import torch


class ImagePreprocessor:
    def __init__(self, transform):
        self.transform = transform

    def preprocess(self, img):
        return self.transform(img)

    def preprocess_batch(self, img_batch):
        return torch.stack([self.preprocess(img) for img in img_batch])


def lift_transform(transform):
    def apply(examples):

        if "image" in examples:
            key = "image"
        elif "img" in examples:
            key = "img"
        else:
            key = None
        examples[key] = [transform(image) for image in examples[key]]

        return examples

    return apply


def channels_to_last(img: torch.Tensor):
    return img.permute(1, 2, 0).contiguous()
