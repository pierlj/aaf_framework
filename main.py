from .train.train import Trainer
from .eval.eval_script import eval_methods
from .config import cfg
import sys
import torch
import os

os.environ['OMP_NUM_THREADS'] = "4"
torch.backends.cudnn.benchmark = True

#################################################################################

cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
cfg.merge_from_list([
    'OUTPUT_DIR',
    '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_TRIPLET',
    'FEWSHOT.SPLIT_FILE', 'classes_split_heli.txt',
    'FSLOSS.MODES', ['TripletLoss']
])

trainer = Trainer(cfg)
trainer.train()

#################################################################################

cfg.merge_from_file('aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
cfg.merge_from_list([
    'OUTPUT_DIR',
    '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_CLS',
    'FEWSHOT.SPLIT_FILE', 'classes_split_heli.txt',
    'FSLOSS.MODES', ['ClassificationLoss']
])

trainer = Trainer(cfg)
trainer.train()

#################################################################################

cfg.merge_from_file(
    'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
cfg.merge_from_list([
    'OUTPUT_DIR',
    '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_RESIZE_AUG',
    'FEWSHOT.SUPPORT.CROP_MODE', 'RESIZE',
    'AUGMENT.VFLIP_PROBA', 0.5,
    'AUGMENT.BRIGHTNESS', 0.4,
    'AUGMENT.CONTRAST', 0.4,
    'AUGMENT.SATURATION', 0.4,
    'AUGMENT.HUE', 0.1,
    'AUGMENT.CUTOUT_PROBA', 0.5,
    'AUGMENT.RANDOM_CROP_PROBA', 0.5

])

trainer = Trainer(cfg)
trainer.train()

#################################################################################

cfg.merge_from_file(
    'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
cfg.merge_from_list([
    'OUTPUT_DIR',
    '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FCOS_DIOR',
    'FEWSHOT.ENABLED', False,
    'DATASETS.TRAIN', ("dior_train",),
    'DATASETS.TEST', ("dior_test",),
    'DATASETS.VAL', ("dior_val",),
    'MODEL.FCOS.NUM_CLASSES', 21,

])

trainer = Trainer(cfg)
trainer.train()

#################################################################################

cfg.merge_from_file(
    'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
cfg.merge_from_list([
    'OUTPUT_DIR',
    '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FCOS_PASCAL',
    'FEWSHOT.ENABLED', False,
    'DATASETS.TRAIN', ("pascalv_merged_train",),
    'DATASETS.TEST', ("pascalv_merged_test",),
    'DATASETS.VAL', ("pascalv_merged_val",),
    'MODEL.FCOS.NUM_CLASSES', 21,

])

trainer = Trainer(cfg)
trainer.train()

#################################################################################

# cfg.merge_from_file(
#     'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
# cfg.merge_from_list([
#     'OUTPUT_DIR',
#     '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_VFLIP_COLOR_CROP',
#     'AUGMENT.VFLIP_PROBA', 0.5,
#     'AUGMENT.BRIGHTNESS', 0.4,
#     'AUGMENT.CONTRAST', 0.4,
#     'AUGMENT.SATURATION', 0.4,
#     'AUGMENT.HUE', 0.1,
#     'AUGMENT.RANDOM_CROP_PROBA', 0.5
# ])

# trainer = Trainer(cfg)
# trainer.train()

##################################################################################

# cfg.merge_from_file(
#     'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
# cfg.merge_from_list([
#     'OUTPUT_DIR',
#     '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_VFLIP_COLOR_CROP',
#     'AUGMENT.VFLIP_PROBA', 0.5,
#     'AUGMENT.BRIGHTNESS', 0.4,
#     'AUGMENT.CONTRAST', 0.4,
#     'AUGMENT.SATURATION', 0.4,
#     'AUGMENT.HUE', 0.1,
#     'AUGMENT.CUTOUT_PROBA', 0.5
# ])

# trainer = Trainer(cfg)
# trainer.train()


model_names = [
    'DOTA/FRW_NO_SOCCER',
    'DOTA/FRW_128_K1',
    'DOTA/FRW_512_K1', 'DOTA/FRW_DOTA256_256',
    'DOTA/FRW_RESIZE_AUG', 'DOTA/FRW_TRIPLET', 'DOTA/FRW_CLS'
]
eval_methods(model_names)




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