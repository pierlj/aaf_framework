import os
import torch
import numpy as np

from shutil import copyfile

import sys

sys.path.insert(1, '/home/pierre/Documents/PHD')

from aaf_framework.config import cfg
from aaf_framework.data.data_handler import DataHandler
from aaf_framework.modeling.detector.detectors import build_detection_model
from aaf_framework.utils.checkpointer import DetectronCheckpointer
from aaf_framework.eval import Evaluator

device = 'cuda:0'

def eval_methods(model_names):
    base_folder = '/home/pierre/Documents/PHD/Experiments_FSFCOS'
    results_folder = '/home/pierre/Documents/PHD/Notebooks/FCOS/Results_robust'
    shot_eval = [1, 3, 5, 10]
    EVAL_EPISODES = 10

    for model_name in model_names:
        print('Evaluation of {}...'.format(model_name))
        for k_shot in shot_eval:
            print('{} shot(s).'.format(k_shot))
            model_path = os.path.join(base_folder, model_name)
            model_file = "model_final_{}_shot.pth".format(k_shot)

            save_path = os.path.join(results_folder, model_name)

            with open(os.path.join(model_path, 'last_checkpoint'), 'w') as f:
                f.write(model_file)

            cfg.merge_from_file(os.path.join(model_path, 'model_cfg.yaml'))
            cfg.merge_from_list(['FEWSHOT.K_SHOT', k_shot])

            model = build_detection_model(cfg)
            model.to(cfg.MODEL.DEVICE).eval()

            checkpointer = DetectronCheckpointer(cfg, model, save_dir=model_path, testing=True)
            _ = checkpointer.load()

            data_handler = DataHandler(cfg, is_train=False, start_iter=0, eval_all=True, data_source='test')
            # data_handler.is_train = True
            evaluator = Evaluator(model, cfg, data_handler)
            with torch.no_grad():
                res = evaluator.eval_all(n_episode=EVAL_EPISODES, seed=42, verbose=False)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            res.to_csv(os.path.join(save_path, model_file + '.csv'), sep=',', index=None)
            copyfile(os.path.join(model_path, model_file), os.path.join(save_path, model_file))
    print('Done!')