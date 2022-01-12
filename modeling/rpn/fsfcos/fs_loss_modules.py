import torch
from torch.nn import functional as F
from torch import nn

from pytorch_metric_learning import losses

class TripletLoss(object):
    def __init__(self, cfg):
        self.loss = losses.TripletMarginLoss(margin=0.05,
                                             triplets_per_anchor=2)

    def __call__(self, query_features, support, classes):
        support_features, support_targets = support
        
        # TO DO complete here
        labels = []
        support_flatten = []

        for feat in support_features:
            N, C, H, W = feat.shape
            feat = feat.flatten(start_dim=2)
            feat = feat.permute(0,2,1) # channel last
            labels.append(torch.arange(N).to(feat).repeat_interleave(H*W))
            support_flatten.append(feat.flatten(end_dim=1))
        labels = torch.cat(labels)
        support_flatten = torch.cat(support_flatten)
        return self.loss(support_flatten, labels)