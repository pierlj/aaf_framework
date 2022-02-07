import torch
from torch.nn import functional as F
from torch import nn

from pytorch_metric_learning import losses

class TripletLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss = losses.TripletMarginLoss(margin=0.1,
                                             triplets_per_anchor='all')

    def __call__(self, query_features, support, classes):
        support_features, support_targets = support
        K = self.cfg.FEWSHOT.K_SHOT

        # TO DO complete here
        labels = []
        support_flatten = []

        for feat in support_features:
            N, C, H, W = feat.shape
            feat = feat.mean(dim=[-2,-1])
            # feat = feat.permute(0,2,1) # channel last
            labels.append(
                torch.arange(len(classes), device=feat.device).repeat_interleave(K))
            support_flatten.append(feat)
        labels = torch.cat(labels)
        support_flatten = torch.cat(support_flatten)
        return self.loss(support_flatten, labels)

class ClassificationLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()

        self.classifier = nn.Sequential(nn.Linear(cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, cfg.MODEL.FCOS.NUM_CLASSES-1)).to('cuda:0')

    def __call__(self, query_features, support, classes):
        support_features, support_targets = support
        K = self.cfg.FEWSHOT.K_SHOT

        assert K*len(classes) == support_features[0].shape[0], 'Mismatch between N_ways * K_shot and input dim.'
        # TO DO complete here
        labels = []
        support_flatten = []

        for feat in support_features:
            N, C, H, W = feat.shape
            feat = feat.mean(dim=[-2, -1])
            labels.append(torch.tensor(classes, device=feat.device).repeat_interleave(K) - 1) # don't forget 0 is background
            support_flatten.append(feat)

        labels = torch.cat(labels)
        support_flatten = torch.cat(support_flatten)
        support_logits = self.classifier(support_flatten)
        return self.loss(support_logits, labels)


class SimilarityLoss(object):
    def __init__(self, cfg):
        self.loss = nn.MSELoss()


    def __call__(self, query_features, support, classes):
        support_features, support_targets = support

        # TO DO complete here
        labels = []
        support_flatten = []

        for feat in support_features:
            N, C, H, W = feat.shape
            feat = feat.flatten(start_dim=2)
            feat = feat.permute(0, 2, 1)  # channel last
            labels.append(torch.arange(N).to(feat).repeat_interleave(H * W))
            support_flatten.append(feat.flatten(end_dim=1))
        labels = torch.cat(labels)
        support_flatten = torch.cat(support_flatten)

        return self.loss(support_logits, labels)