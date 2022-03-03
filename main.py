from .train.train import Trainer
from .eval.eval_script import eval_methods
from .config import cfg
import sys
import torch
import os

os.environ['OMP_NUM_THREADS'] = "4"
torch.backends.cudnn.benchmark = True


# model_names = [
#     'DOTA/FRW_BASELINE_PASCALV_MERGED', 'DOTA/FRW_BASELINE_DOTA',
#     'DOTA/FRW_BASELINE_DIOR'
# ]
# eval_methods(model_names)

#################################################################################

cfg.merge_from_file(
    'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
cfg.merge_from_list([
    'OUTPUT_DIR', '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/PFRCN_DOTA',
    'FEWSHOT.SUPPORT.CROP_MODE', 'RESIZE',
    'AUGMENT.VFLIP_PROBA', 0.5,
    'AUGMENT.BRIGHTNESS', 0.4,
    'AUGMENT.CONTRAST', 0.4,
    'AUGMENT.SATURATION', 0.4,
    'AUGMENT.HUE', 0.1,
    'AUGMENT.CUTOUT_PROBA', 0.5,
    'AUGMENT.RANDOM_CROP_PROBA', 0.5,
    'FEWSHOT.K_SHOT', 1,
    'SOLVER.IMS_PER_BATCH', 8,
    'FEWSHOT.AAF.CFG', '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/identity.yaml',
    'FEWSHOT.USE_PROTO_CLASSIFIER', True,
    'DATASETS.TRAIN', ("dota_train", ),
    'DATASETS.TEST', ("dota_test", ),
    'DATASETS.VAL', ("dota_val", ),
    # 'MODEL.FCOS.NUM_CLASSES', 21,
    'FEWSHOT.EPISODES', 1500,
    'FINETUNE.EPISODES', 15000,
    # 'INPUT.MIN_SIZE_TRAIN', (800,),
    # 'INPUT.MAX_SIZE_TRAIN', 800,
    # 'INPUT.MIN_SIZE_TEST', 800,
    # 'INPUT.MAX_SIZE_TEST', 800,
    'RANDOM.SEED', 2048,
])

trainer = Trainer(cfg)
trainer.train()

# #################################################################################

# cfg.merge_from_file(
#     'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
# cfg.merge_from_list([
#     'OUTPUT_DIR',
#     '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/WSAAN_RESIZE_AUG_DIOR_2',
#     'FEWSHOT.SUPPORT.CROP_MODE', 'RESIZE',
#     'AUGMENT.VFLIP_PROBA', 0.5,
#     'AUGMENT.BRIGHTNESS', 0.4,
#     'AUGMENT.CONTRAST', 0.4,
#     'AUGMENT.SATURATION', 0.4,
#     'AUGMENT.HUE', 0.1,
#     'AUGMENT.CUTOUT_PROBA', 0.5,
#     'AUGMENT.RANDOM_CROP_PROBA', 0.5,
#     'FEWSHOT.K_SHOT', 1,
#     'SOLVER.IMS_PER_BATCH', 8,
#     'FEWSHOT.AAF.CFG', '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/self_adapt.yaml',
#     'DATASETS.TRAIN', ("dior_train",),
#     'DATASETS.TEST', ("dior_test",),
#     'DATASETS.VAL', ("dior_val",),
#     # 'MODEL.FCOS.NUM_CLASSES', 21,
#     'FEWSHOT.EPISODES', 1500,
#     'FINETUNE.EPISODES', 15000,
#     # 'INPUT.MIN_SIZE_TRAIN', (800,),
#     # 'INPUT.MAX_SIZE_TRAIN', 800,
#     # 'INPUT.MIN_SIZE_TEST', 800,
#     # 'INPUT.MAX_SIZE_TEST', 800,
#     'RANDOM.SEED', 2048,

# ])

# trainer = Trainer(cfg)
# trainer.train()

#################################################################################

# cfg.merge_from_file(
#     'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
# cfg.merge_from_list([
#     'OUTPUT_DIR',
#     '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/FRW_RESIZE_AUG_PASCAL',
#     'FEWSHOT.SUPPORT.CROP_MODE', 'RESIZE',
#     'AUGMENT.VFLIP_PROBA', 0.0,
#     'AUGMENT.BRIGHTNESS', 0.4,
#     'AUGMENT.CONTRAST', 0.4,
#     'AUGMENT.SATURATION', 0.4,
#     'AUGMENT.HUE', 0.1,
#     'AUGMENT.CUTOUT_PROBA', 0.5,
#     'AUGMENT.RANDOM_CROP_PROBA', 0.5,
#     'FEWSHOT.K_SHOT', 1,
#     'SOLVER.IMS_PER_BATCH', 8,
#     'DATASETS.TRAIN', ("pascalv_merged_train",),
#     'DATASETS.TEST', ("pascalv_merged_test",),
#     'DATASETS.VAL', ("pascalv_merged_val",),
#     'MODEL.FCOS.NUM_CLASSES', 21,
#     # 'FEWSHOT.EPISODES', 1500,
#     # 'FINETUNE.EPISODES', 15000,

# ])

# trainer = Trainer(cfg)
# trainer.train()

# #################################################################################

# cfg.merge_from_file(
#     'aaf_framework/config_files/fcos_R_50_FPN_DOTA_baseline.yaml')
# cfg.merge_from_list([
#     'OUTPUT_DIR',
#     '/home/pierre/Documents/PHD/Experiments_FSFCOS/DOTA/WSAAN_BASELINE_PASCAL_2',

#     'SOLVER.IMS_PER_BATCH', 8,
#     'FEWSHOT.AAF.CFG', '/home/pierre/Documents/PHD/aaf_framework/config_files/aaf_module/self_adapt.yaml',
#     'DATASETS.TRAIN', ("pascalv_merged_train",),
#     'DATASETS.TEST', ("pascalv_merged_test",),
#     'DATASETS.VAL', ("pascalv_merged_val",),
#     'MODEL.FCOS.NUM_CLASSES', 21,
#     'FEWSHOT.EPISODES', 1500,
#     # 'FINETUNE.EPISODES', 15000,

# ])

# trainer = Trainer(cfg)
# trainer.train()

#################################################################################

model_names = [
    'DOTA/DANA_RESIZE_AUG_DIOR',
    # 'DOTA/WSAAN_RESIZE_AUG_DIOR_2',
    # 'DOTA/WSAAN_BASELINE_PASCAL_2',
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