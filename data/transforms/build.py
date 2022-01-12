# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True, is_support=False):

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN,
                                      std=cfg.INPUT.PIXEL_STD,
                                      to_bgr255=to_bgr255)


    if is_train:
        if cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0] == -1:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
        else:
            assert len(cfg.INPUT.MIN_SIZE_RANGE_TRAIN) == 2, \
                "MIN_SIZE_RANGE_TRAIN must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[0],
                cfg.INPUT.MIN_SIZE_RANGE_TRAIN[1] + 1
            ))
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = cfg.AUGMENT.FLIP_PROBA  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0.0

    transform_list = [
        T.Resize(min_size, max_size),
        T.RandomHorizontalFlip(flip_prob),
        T.RandomVerticalFlip(flip_prob),
    ]

    if is_train and not is_support:
        transform_list.append(
            T.ColorJitter(brightness=cfg.AUGMENT.BRIGHTNESS,
                      contrast=cfg.AUGMENT.CONTRAST,
                      saturation=cfg.AUGMENT.SATURATION,
                      hue=cfg.AUGMENT.HUE)
                            )

    transform_list = transform_list + [T.ToTensor(), normalize_transform]

    if is_train and not is_support:
        transform_list.append(T.Cutout(cfg.AUGMENT.CUTOUT_PROBA, cfg.AUGMENT.CUTOUT_SCALE))

    transform = T.Compose(transform_list)
    
    return transform
