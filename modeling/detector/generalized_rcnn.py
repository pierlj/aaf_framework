# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from copy import deepcopy

from fcos_core.structures.image_list import to_image_list

from fcos_core.modeling.backbone import build_backbone
from ..rpn.rpn import build_rpn
from fcos_core.modeling.roi_heads.roi_heads import build_roi_heads
from ..support_extractor import ReweightingModule, MSReweightingModule
from ...utils.visualization import plot_img_only


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.cfg = cfg

    def forward(self, images, targets=None, classes=None, support=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)

        if self.cfg.AMP.ACTIVATED and self.cfg.AMP.MODE == 'O3':
            images.tensors = images.tensors.half()

        features = self.backbone(images.tensors)[:self.cfg.FEWSHOT.FEATURE_LEVEL]
        proposals, proposal_losses = self.rpn(images, features, targets, classes=classes, support=support)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result


class FSGeneralizedRCNN(GeneralizedRCNN):
    """
    Add a method on top of GeneralizedRCNN to compute support features.
    """
    def __init__(self, cfg):
        super(FSGeneralizedRCNN, self).__init__(cfg)
        device = torch.device(cfg.MODEL.DEVICE)
        if cfg.FEWSHOT.SUPPORT_EXTRACTOR == 'same':
            self.support_features_extractor = self.backbone
        elif cfg.FEWSHOT.SUPPORT_EXTRACTOR == 'multiscale_distinct':
            self.support_features_extractor = MSReweightingModule().to(device)
        elif cfg.FEWSHOT.SUPPORT_EXTRACTOR == 'siamese':
            self.support_features_extractor = build_backbone(cfg)
        else:
            self.support_features_extractor = ReweightingModule().to(device)


    def compute_support_features(self, support_loader, device):
        support_features = []
        support_targets = []
        self.support_examples__ = []
        self.support_targets__ = []
        L = self.cfg.FEWSHOT.FEATURE_LEVEL

        for idx, (images, targets, _) in enumerate(support_loader):
            # display support images
            # plot_img_only(images.tensors[0], self.rpn.head.aaf_module.cfg)
            # print(images.tensors.shape)
            if self.cfg.FEWSHOT.SUPPORT.CROP_MODE == 'MULTI_SCALE':
                N, C, H, W = images.tensors.shape
                images.tensors = images.tensors.reshape(3*N, C//3, H, W)
                # targets = [ms_target for target in targets for ms_target in target.get_field('ms_targets')]
                self.support_examples__.append(images.tensors[2::3].clone())
            else:
                self.support_examples__.append(images.tensors.clone())

            self.support_targets__.append(targets)
            images = images.to(device, non_blocking=True)
            targets = [target.to(device) for target in targets]
            images = to_image_list(images)

            if self.cfg.AMP.ACTIVATED and self.cfg.AMP.MODE == 'O3':
                images.tensors = images.tensors.half()

            features = self.support_features_extractor(images.tensors)[:L]
            feat_ch = features[0].shape[1]
            if self.cfg.FEWSHOT.SUPPORT.CROP_MODE == 'MULTI_SCALE':
                # features = [
                #     feat.reshape(N, 3, feat_ch, H // (8 * 2**l),
                #                  W // (8 * 2**l))[:, l]
                #     for l, feat in enumerate(features)
                # ]
                features_masked = []
                for l, feat in enumerate(features):
                    ms_targets = [t for target in targets for t in target.get_field('ms_targets')]
                    masks = torch.zeros_like(feat)
                    for i, ms_target in enumerate(ms_targets):
                        bbox = ms_target.resize(feat.shape[-2:]).bbox[0]
                        bbox = bbox.clamp(0, feat.shape[-1])
                        bbox[:2] = bbox[:2].floor()
                        bbox[2:] = bbox[2:].ceil()
                        bbox = bbox.long()
                        masks[i, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
                    feat = feat.reshape(N, 3, feat_ch, *feat.shape[-2:])
                    masks = masks.reshape(N, 3, feat_ch, *feat.shape[-2:])
                    feat = feat * masks
                    feat = feat.sum(dim=[-1,-2], keepdim=True) / masks.sum(dim=[-1,-2], keepdim=True)
                    # features_masked.append(feat.mean(dim=1)) # average multiple scale
                    features_masked.append(feat[:, l]) # keep only corresponding level
                features = features_masked

            support_features.append(features)
            support_targets = support_targets + targets

        support_features = [
            torch.cat([features[l] for features in support_features])
                for l in range(len(support_features[0]))
        ]

        return support_features, support_targets
