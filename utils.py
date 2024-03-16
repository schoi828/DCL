# -*- coding: utf-8 -*-
import os
import logging
id_fine= {
        0: 'apple',
        1: 'aquarium_fish',
        2: 'baby',
        3: 'bear',
        4: 'beaver',
        5: 'bed',
        6: 'bee',
        7: 'beetle',
        8: 'bicycle',
        9: 'bottle',
        10: 'bowl',
        11: 'boy',
        12: 'bridge',
        13: 'bus',
        14: 'butterfly',
        15: 'camel',
        16: 'can',
        17: 'castle',
        18: 'caterpillar',
        19: 'cattle',
        20: 'chair',
        21: 'chimpanzee',
        22: 'clock',
        23: 'cloud',
        24: 'cockroach',
        25: 'couch',
        26: 'crab',
        27: 'crocodile',
        28: 'cup',
        29: 'dinosaur',
        30: 'dolphin',
        31: 'elephant',
        32: 'flatfish',
        33: 'forest',
        34: 'fox',
        35: 'girl',
        36: 'hamster',
        37: 'house',
        38: 'kangaroo',
        39: 'computer_keyboard',
        40: 'lamp',
        41: 'lawn_mower',
        42: 'leopard',
        43: 'lion',
        44: 'lizard',
        45: 'lobster',
        46: 'man',
        47: 'maple_tree',
        48: 'motorcycle',
        49: 'mountain',
        50: 'mouse',
        51: 'mushroom',
        52: 'oak_tree',
        53: 'orange',
        54: 'orchid',
        55: 'otter',
        56: 'palm_tree',
        57: 'pear',
        58: 'pickup_truck',
        59: 'pine_tree',
        60: 'plain',
        61: 'plate',
        62: 'poppy',
        63: 'porcupine',
        64: 'possum',
        65: 'rabbit',
        66: 'raccoon',
        67: 'ray',
        68: 'road',
        69: 'rocket',
        70: 'rose',
        71: 'sea',
        72: 'seal',
        73: 'shark',
        74: 'shrew',
        75: 'skunk',
        76: 'skyscraper',
        77: 'snail',
        78: 'snake',
        79: 'spider',
        80: 'squirrel',
        81: 'streetcar',
        82: 'sunflower',
        83: 'sweet_pepper',
        84: 'table',
        85: 'tank',
        86: 'telephone',
        87: 'television',
        88: 'tiger',
        89: 'tractor',
        90: 'train',
        91: 'trout',
        92: 'tulip',
        93: 'turtle',
        94: 'wardrobe',
        95: 'whale',
        96: 'willow_tree',
        97: 'wolf',
        98: 'woman',
        99: 'worm'}

id_coarse= {
        0: 'aquatic mammals',
        1: 'fish',
        2: 'flowers',
        3: 'food containers',
        4: 'fruit and vegetables',
        5: 'household electrical device',
        6: 'household furniture',
        7: 'insects',
        8: 'large carnivores',
        9: 'large man-made outdoor things',
        10: 'large natural outdoor scenes',
        11: 'large omnivores and herbivores',
        12: 'medium-sized mammals',
        13: 'non-insect invertebrates',
        14: 'people',
        15: 'reptiles',
        16: 'small mammals',
        17: 'trees',
        18: 'vehicles 1',
        19: 'vehicles 2'}

def logger_setting(exp_name, save_dir):
    logger = logging.getLogger(exp_name)
    # formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    formatter = logging.Formatter('[%(asctime)s] : %(message)s')

    log_out = os.path.join(save_dir, exp_name, 'train.log')
    file_handler = logging.FileHandler(log_out)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)

    return logger

def get_str_labels(dataset):
    out = []
    if dataset=='CIFAR10':
        out = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset=='CIFAR100':
        for i in id_fine:
            out.append(id_fine[i])
    elif dataset =='CIFAR100_C':
        for i in id_coarse:
            out.append(id_coarse[i])
    elif dataset =='MNIST':
        for i in range(10):
            out.append(str(i))
    return out