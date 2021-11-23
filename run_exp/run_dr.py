from ..train.train import Trainer
from FCOS.fcos_core.config import cfg

cfg.merge_from_file('fsfcos/cfg/fcos_R_50_FPN_VOC.yaml')
cfg.merge_from_list([
    'FEWSHOT.AAF.CFG',
    '/home/pierre/Documents/PHD/fsfcos/cfg/aaf_module/dynamic.yaml',
    'OUTPUT_DIR',
    '/home/pierre/Documents/PHD/Experiments_FSFCOS/Experiments_paper/DR'
])

trainer = Trainer(cfg)
trainer.train()
