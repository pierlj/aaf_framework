from .train.train import Trainer
from .config import cfg
import sys
import torch


# cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_VHR.yaml')
# cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_COCO.yaml')
# cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA.yaml')
# cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_VOC.yaml')

# trainer = Trainer(cfg)
# trainer.train()

# del trainer

# cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA.yaml')
# cfg.merge_from_list([
#     'OUTPUT_DIR',
#     '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_256_BB_FREEZE_2',
#     'FINETUNE.FREEZE_AT', 2,
# ])

# trainer = Trainer(cfg)
# trainer.train()

# for crop_method in ['RESIZE', 'REFLECT']:
#     cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA.yaml')
#     cfg.merge_from_list([
#         'OUTPUT_DIR',
#         '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_256_{}'.format(
#             crop_method), 'FEWSHOT.SUPPORT.CROP_MODE', crop_method,
#         'SOLVER.IMS_PER_BATCH', 16
#     ])
#     trainer = Trainer(cfg)
#     trainer.train()
#     del trainer

for n_ch in [128, 512]:
    cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA.yaml')
    cfg.merge_from_list([
        'OUTPUT_DIR',
        '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_{}_K1'.format(
            n_ch), 'MODEL.RESNETS.BACKBONE_OUT_CHANNELS', n_ch,
        'SOLVER.IMS_PER_BATCH', 8
    ])
    trainer = Trainer(cfg)
    trainer.train()
    del trainer

cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA.yaml')
cfg.merge_from_list([
    'OUTPUT_DIR',
    '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_256_CROP_REISZE',
    'AUGMENT.RANDOM_CROP_PROBA', 0.5,
])

trainer = Trainer(cfg)
trainer.train()
del trainer


# method_dict = {
#     'FRW': '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/feature_reweighting.yaml',
#     'DANA': '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/dana.yaml',
#     'META_FASTER': '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/meta_faster_rcnn.yaml',
#     'DRL': '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/dynamic.yaml',
#     'WSAAN': '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/self_adapt.yaml',
# }

# for method in ['WSAAN', 'DANA',   ]:
#     cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA.yaml')
#     cfg.merge_from_list([
#         'OUTPUT_DIR',
#         '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/{}_256'.format(
#             method), 'FEWSHOT.AAF.CFG', method_dict[method],
#     ])
#     trainer = Trainer(cfg)
#     trainer.train()
#     del trainer