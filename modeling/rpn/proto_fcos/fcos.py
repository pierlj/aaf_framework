import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d

from ...aaf import AAFModule


class ProtoFCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Detection Head for Few-shot FCOS

        Arguments:
            - cfg: config object of the network
            - in_channels (int): number of channels of the input feature
        """
        super(ProtoFCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        self.aaf_module = AAFModule(cfg)
        aaf_channels = self.aaf_module.aaf_cfg.OUT_CH

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            if i != 0:
                aaf_channels = in_channels

            cls_tower.append(
                conv_func(
                    aaf_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    aaf_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = ProtoClassifier()


        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, query_features, support, classes_episode):

        support_features, support_targets = support
        query_features_processed = self.aaf_module(query_features,
                                                   support_features,
                                                   support_targets)

        logits = {c: [] for c in classes_episode}
        bbox_reg = {c: [] for c in classes_episode}
        centerness = {c: [] for c in classes_episode}

        # Retrieve support features from aaf combine per class
        # and re arrange as dict of prototypes
        prototypes = self.compute_prototypes(support_features, classes_episode)

        for current_class, pyramid_feat in query_features_processed.items():
            for l, feature in enumerate(pyramid_feat):
                cls_tower = self.cls_tower(feature)
                box_tower = self.bbox_tower(feature)

                logits[current_class].append(self.cls_logits(feature,
                                                        prototypes[current_class][l],
                                                        classes_episode))
                if self.centerness_on_reg:
                    centerness[current_class].append(self.centerness(box_tower))
                else:
                    centerness[current_class].append(self.centerness(cls_tower))

                bbox_pred = self.scales[l](self.bbox_pred(box_tower))
                if self.norm_reg_targets:
                    bbox_pred = F.relu(bbox_pred)
                    if self.training:
                        bbox_reg[current_class].append(bbox_pred)
                    else:
                        bbox_reg[current_class].append(bbox_pred * self.fpn_strides[l])
                else:
                    bbox_reg[current_class].append(torch.exp(bbox_pred))

        return logits, bbox_reg, centerness

    def compute_prototypes(self, support_features, classes):
        N, C, _, _ = support_features[0].shape
        K = self.aaf_module.cfg.FEWSHOT.K_SHOT
        prototypes = [feat.mean(dim=[-1,-2]).reshape(N // K, K, C)
                            for feat in support_features]

        proto_dict = {c: [proto[idx] for proto in prototypes] for idx, c in enumerate(classes)}
        return proto_dict


class ProtoClassifier(torch.nn.Module):
    def __init__(self):
        super(ProtoClassifier, self).__init__()

        self.average_proto = False
        self.sigma = 0.1

    def forward(self, query_features, prototypes, classes):

        # protos K, C
        if self.average_proto:
            prototypes = prototypes.mean(dim=0, keepdim=True)
        feature_vectors = query_features #B, C, H, W
        feature_vectors = feature_vectors.permute(0,2,3,1)
        B, H, W, C = feature_vectors.shape
        feature_vectors = feature_vectors.reshape(B, -1, C).contiguous()
        prototypes = prototypes.repeat(B, 1, 1)
        feature_vectors = feature_vectors / feature_vectors.norm(dim=-1,
                                                                 keepdim=True)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

        distance_matrix = torch.cdist(
            prototypes, feature_vectors)  #B, K or 1 , HW

        K = prototypes.shape[1]
        dist_closest_proto = distance_matrix.reshape(B, K, H*W).min(dim=1)[0]
        cls_level_logits = -dist_closest_proto / 2 / self.sigma**2
        cls_level_logits = cls_level_logits.reshape(B, 1, H, W)

        return cls_level_logits


class ProtoFCOSModule(torch.nn.Module):
    """
    Module for few-shot FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(ProtoFCOSModule, self).__init__()

        head = ProtoFCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES


    def forward(self, images, features, targets=None, classes=None, support=None):
        """
        Arguments:
            -images (ImageList): images for which we want to compute the predictions
            -features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            -targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        box_cls, box_regression, centerness = self.head(features, support, classes)
        locations = self.compute_locations(features)
        self.classes = classes

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations