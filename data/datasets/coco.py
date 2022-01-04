# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import os
import json
import numpy as np
from collections import defaultdict

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints

from ...utils.visualization import plot_single_img_boxes
from .cropping import CroppingModule

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset_(torchvision.datasets.CocoDetection):
    """
    COCODataset wrapper on torchvision class from FCOS. 
    """
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset_, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):

        img, anno = super(COCODataset_, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

class COCODataset(COCODataset_):
    """
    Our COCODataset wrapper on the one written for FCOS. 

    It samples classes from selected_classes to allow episodic training with 
    subsets of classes only.
    """
    def __init__(self, *args, selected_classes=None, is_support=None, filter_obj=True, **kwargs):
        super(COCODataset, self).__init__(*args, **kwargs)
        self.selected_classes = selected_classes
        self.is_support = is_support

        self.class_table = {self.json_category_id_to_contiguous_id[k]: list(set(class_idx_list))
                            for k, class_idx_list in self.coco.catToImgs.items()}

        if filter_obj:
            self.filter_class_table()

        self.image_ids_to_ids = {img_id: dataset_id for dataset_id, img_id in enumerate(self.ids)}


    def __getitem__(self, idx):
        img, target, idx = super(COCODataset, self).__getitem__(idx)

        # When fewshot is enabled filter out classes not in the episode classes
        if self.selected_classes is not None:
            labels = target.get_field('labels')
            bbox = target.bbox
            keep = torch.nonzero((labels.unsqueeze(-1) == self.selected_classes).sum(dim=-1),
                                 as_tuple=False).view(-1)

            labels = labels[keep]
            bbox = bbox[keep]

            target.bbox = bbox
            target.add_field('labels', labels)
            # plot_single_img_boxes(img, target, self.cfg)

        return img, target, idx

    def filter_class_table(self):
        AREA_TH = 0
        inf = np.inf

        imgToAnns = {}
        catToImgs = defaultdict(list)

        for img_id in self.coco.imgToAnns:
            img_annot = self.coco.getAnnIds(imgIds=[img_id],
                                            areaRng=[AREA_TH, inf])
            if img_annot != []:
                imgToAnns[img_id] = [
                    self.coco.anns[annot_id] for annot_id in img_annot
                ]
        self.coco.imgToAnns = imgToAnns

        for k, img_list in self.class_table.items():
            self.class_table[k] = [
                img_id for img_id in img_list if self.coco.getAnnIds(
                    imgIds=[img_id],
                    catIds=[self.contiguous_category_id_to_json_id[k]],
                    areaRng=[AREA_TH, inf]) != []
            ]
            # catToImgs[self.contiguous_category_id_to_json_id[k]] = [img_id]
        self.coco.anns = {
            annot_id: self.coco.anns[annot_id]
            for annot_id in self.coco.getAnnIds(areaRng=[AREA_TH, inf])
        }
        for ann in self.coco.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

        self.coco.catToImgs = catToImgs


class SupportCOCODataset(COCODataset):
    """
    Another wrapper on COCODataset.

    This one is specifically designed for support set. 
    It keeps only one annotation per image and modify the image and the annotation 
    according to its cropping module. 
    """
    def __init__(self, *args, rng=None, **kwargs):
        self.cfg = kwargs['cfg']
        del kwargs['cfg']
        # kwargs['filter_obj'] = False
        super(SupportCOCODataset, self).__init__(*args, **kwargs)

        self.crop = self.cfg.FEWSHOT.SUPPORT.CROP
        self.fixed_size = self.cfg.FEWSHOT.SUPPORT.CROP_SIZE
        self.margin = self.cfg.FEWSHOT.SUPPORT.CROP_MARGIN

        self.filter_class_table()

        self.support_cropper = CroppingModule(self.cfg,
                                              self.cfg.FEWSHOT.SUPPORT.CROP_MODE)

        if rng is not None:
            self.rng = rng
        else:
            self.rng = torch.Generator()

    def __getitem__(self, item_idx):
        idx, class_selected = item_idx
        self.selected_classes = torch.Tensor([class_selected])

        img, target, idx = super(SupportCOCODataset, self).__getitem__(idx)
        # Sample only 1 example per image
        labels = target.get_field('labels')
        bbox = target.bbox
        keep = torch.randint(labels.shape[0], (1,), generator=self.rng)

        wh = bbox[:,2:] - bbox[:,:2]
        areas = wh[:,0] * wh[:,1]
        keep = areas.argmax().unsqueeze(0)

        labels = labels[keep]
        bbox = bbox[keep]

        target.bbox = bbox
        target.add_field('labels', labels)
        if self.crop:
            # plot_single_img_boxes(img, target, self.cfg)
            img, target = self.support_cropper.crop(img, target)
            # plot_single_img_boxes(img, target, self.cfg)

        return img, target, idx



class FinetuneCOCODataset(COCODataset):
    def __init__(self, *args, rng=None, **kwargs):
        self.cfg = kwargs['cfg']
        del kwargs['cfg']
        super(FinetuneCOCODataset, self).__init__(*args, **kwargs)

        if rng is not None:
            self.rng = rng
        else:
            self.rng = torch.Generator()

    def __getitem__(self, item_idx):
        idx, class_selected = item_idx
        self.selected_classes = torch.Tensor([class_selected])
        img, target, idx = super(FinetuneCOCODataset, self).__getitem__(idx)

        # Sample only 1 example per image
        labels = target.get_field('labels')
        bbox = target.bbox
        keep = torch.randint(labels.shape[0], (1, ), generator=self.rng)

        labels = labels[keep]
        bbox = bbox[keep]

        target.bbox = bbox
        target.add_field('labels', labels)

        # plot_single_img_boxes(img, target, self.cfg)
        return img, target, idx