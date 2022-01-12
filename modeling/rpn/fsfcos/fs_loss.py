import torch
from torch.nn import functional as F
from torch import nn

from . import fs_loss_modules as L

class FSLossComputation(object):
    """
    This class computes the few-shot losses.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_modes = cfg.FSLOSS.MODES

        self.loss_modules = {}
        for mode in self.loss_modes:
            self.loss_modules[mode]= getattr(L, mode)(cfg)


    def __call__(self, query_features, classes, support):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (dict[list[Tensor]])
            box_regression (dict[list[Tensor]])
            centerness (dict[list[Tensor]])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        losses = {}
        for name, loss_evaluator in self.loss_modules.items():
            losses[name] = loss_evaluator(query_features, classes, support)

        return losses
