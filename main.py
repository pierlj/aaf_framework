from .train.train import Trainer
from .config import cfg
import sys
import torch

cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_VOC_PROTO.yaml')


trainer = Trainer(cfg)
trainer.train()
