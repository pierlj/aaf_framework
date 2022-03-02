# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist, boxlist_iou


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        elif target is None:
            return image
        else:
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):

        self.color_jitter = torchvision.transforms.ColorJitter(brightness=brightness,
                                                               contrast=contrast,
                                                               saturation=saturation,
                                                               hue=hue)

    def __call__(self, image, target=None):
        image = self.color_jitter(image)

        if target is None:
            return image
        return image, target


class Cutout(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), level='image'):

        self.cutout = torchvision.transforms.RandomErasing(p=p,
                                                           scale=scale,
                                                           ratio=(0.3, 3.3),
                                                           value=0,
                                                           inplace=False)

        self.level = level

    def __call__(self, image, target=None):
        if target is None:
            return image

        if self.level == 'image':
            image = self.cutout(image)
        elif self.level == 'object':
            image = image.clone()
            for bbox in target.bbox:
                bbox = bbox.long()
                cropped_image = image[..., bbox[1]:bbox[3], bbox[0]:bbox[2]]
                cropped_image = self.cutout(cropped_image)
                image[..., bbox[1]:bbox[3], bbox[0]:bbox[2]] = cropped_image
        return image, target


class RandomResizeCrop(object):
    def __init__(self, size, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, image, target=None):
        if target is not None:
            if random.random() < self.p:
                if 'masks' in target.fields():
                    del target.extra_fields['masks']
                selected_indices = self.random_subset_selection(len(target))
                selected_target = cat_boxlist(
                    [target[idx:(idx + 1)] for idx in selected_indices])

                overall_box = self.compute_overall_bbox(selected_target)

                delta = (0.25 * torch.randn(4) + 0.5).clamp(0, 1)

                cropping_box = overall_box + (torch.tensor(
                    [0, 0, *image.shape[-2:][::-1]]) - overall_box) * delta
                cropping_box = cropping_box.long()
                cropping_box = self.square_box(cropping_box, image.shape[-2:])

                over_box_list = BoxList(cropping_box.unsqueeze(0),
                                        image.shape[-2:][::-1])

                iou_with_target = boxlist_iou(over_box_list, target)[0]
                selected_indices_all = torch.nonzero(
                    iou_with_target >= 0.5 * target.area() /
                    over_box_list.area()).flatten()
                selected_target = cat_boxlist(
                    [target[idx:(idx + 1)] for idx in selected_indices_all])

                cropped_img = image[..., cropping_box[1]:cropping_box[3],
                                    cropping_box[0]:cropping_box[2]]
                selected_target = selected_target.crop(cropping_box)

                cropped_img = torch.nn.functional.interpolate(
                    cropped_img.unsqueeze(0), self.size)[0]
                selected_target = selected_target.resize(self.size)
                return cropped_img, selected_target
            return image, target

        else:
            return image

    def random_subset_selection(self, n_boxes):
        k = random.randint(1, min(2, n_boxes))
        subset = random.sample([i for i in range(n_boxes)], k)
        return subset

    def compute_overall_bbox(self, bbox_list):
        return torch.cat([
            bbox_list.bbox[:, :2].min(dim=0)[0],
            bbox_list.bbox[:, 2:].max(dim=0)[0]
        ])

    def square_box(self, box, img_shape):
        hw = box[2:] - box[:2]
        max_dim = torch.ones_like(hw) * hw.max()

        enlarged_box = box.clone()
        enlarged_box[:2] = enlarged_box[:2] - (max_dim - hw) / 2
        enlarged_box[2:] = enlarged_box[2:] + (max_dim - hw) / 2

        enlarged_box[::2] = enlarged_box[::2].clamp(0, img_shape[1])
        enlarged_box[1::2] = enlarged_box[1::2].clamp(0, img_shape[0])
        return enlarged_box
