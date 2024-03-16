import torch
import torch.nn as nn
from einops import repeat, rearrange
import torch.nn.functional as F
import numpy as np

def similarity_matrix(x, no_std=False):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not no_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R

class Normalize(nn.Module):
    def __init__(self, dim = 1, power=2):
        super().__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

#Dictionary Contrastive Loss
def emb_pred_loss(latents,labels,emb_dict, patch=0,temperature=1,perturb=False,norm=False):
    
    if norm:
        emb_dict = F.normalize(emb_dict,dim=1)
        latents = F.normalize(latents,dim=1)
    if len(latents.shape) == 4:
        latents = rearrange(latents,"b c h w -> (h w) b c") #latents.flatten(2).permute(2,0,1) #seq, b, ch
    elif len(latents.shape) == 3:
        latents = rearrange(latents,"b c p -> p b c") #latents.flatten(2).permute(2,0,1) #seq, b, ch
    elif patch>0:
        latents = rearrange(latents,"b (p c) -> p b c", p=patch)
    else:
        latents = latents.unsqueeze(1)
    if emb_dict.shape[-1] != latents.shape[-1]:
        emb_dict = F.adaptive_avg_pool1d(emb_dict, latents.shape[-1])

    if len(emb_dict.shape) < 3:
        pred = torch.matmul(latents,emb_dict.T).mean(0)
    else:
        pred = torch.matmul(latents,emb_dict.transpose(1,2)).squeeze(1)
    pred/=temperature
    pred_loss = 0
    if labels is not None:
        pred_loss = F.cross_entropy(pred, labels)
       
    return pred, pred_loss


"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, temperature=0.07, mean=True,mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        if len(features.shape)>2:
            if mean:
                features = features.flatten(2).mean(-1)#b, ch, h*w -> b, ch
            else:
                features = features.flatten(1)
        features = F.normalize(features, dim=1)
        features = features.unsqueeze(1)#b,1,ch

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1) #b, 1, ch

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] #1
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #b, ch
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
