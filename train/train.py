import argparse
import math
from tqdm import tqdm

from torch._C import Argument
import os
import datetime
import logging
import time
import sys

import torch
from fcos_core.config import cfg
from fcos_core.utils.metric_logger import MetricLogger
from fcos_core.engine.trainer import reduce_loss_dict
from fcos_core.engine.trainer import do_train

from ..modeling.detector import build_detection_model
from .utils import make_lr_scheduler, make_optimizer
from ..data.data_handler import DataHandler
from ..utils.checkpointer import DetectronCheckpointer
from ..utils.visualization import plot_img_boxes
from ..eval import Evaluator
from ..utils.custom_logger import CustomLogger


class Trainer():
    """
    Trainer object that manages the training of all networks no matter
    if it few-shot or not (or finetuning)

    Builds network and environment from cfg file. 
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Model and optimizer construction
        self.model = build_detection_model(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        self.optimizer = make_optimizer(cfg, self.model)
        self.scheduler = make_lr_scheduler(cfg, self.optimizer)

        self.arguments = {}
        self.arguments["iteration"] = 0

        self.episodes = cfg.FEWSHOT.EPISODES
        self.logging_int = cfg.LOGGING.INTERVAL
        self.logging_eval_int = cfg.LOGGING.EVAL_INTERVAL
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

        self.fintuning_start_iter = 0


        # check if output_dir exists else create it
        output_dir = cfg.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        save_to_disk = True
        self.checkpointer = DetectronCheckpointer(
            cfg, self.model, self.optimizer, self.scheduler, output_dir, save_to_disk
        )

        extra_checkpoint_data = self.checkpointer.load(self.cfg.MODEL.WEIGHT)
        self.arguments.update(extra_checkpoint_data)

        # by default is_finetuning is false it will be set to true when it starts
        # if only finetuning then the number of base training is 0 but finetuning will
        # be set later anyway.
        self.is_finetuning = False

        # Main data object for base training
        self.data_handler = DataHandler(cfg,
                                        base_classes=not self.is_finetuning,
                                        is_train=True,
                                        start_iter=self.arguments['iteration'],
                                        is_finetune=self.is_finetuning)

        self.tensorboard = CustomLogger(log_dir='/home/pierre/Documents/PHD/Experiments/logs',
                                        notify=False)

        self.evaluator_test = None
        self.evaluator_train = None

    def train(self):
        """
        Main training loop. Starts base training and multiple finetunings
        with different number of shots after (if finetuning is enabled). 
        """
        self.save_config()
        if self.cfg.FEWSHOT.ENABLED:
            self.query_loader, self.support_loader, self.train_classes = self.data_handler.get_dataloader()
            self.do_fs_train()
            
            if self.cfg.FINETUNING:
                self.is_finetuning = True
                self.fintuning_start_iter = self.max_iter
                for k_shot in self.cfg.FINETUNE.SHOTS:
                    # number of episodes specified in cfg finetune is for the
                    # 1 shot case, number is adjusted to have the same number of
                    # updates with each shots.
                    episodes = self.cfg.FINETUNE.EPISODES // math.ceil(
                        self.cfg.FEWSHOT.N_WAYS_TRAIN /
                        self.cfg.SOLVER.IMS_PER_BATCH * k_shot)
                    self.prepare_finetuning(k_shot, episodes)
                    self.query_loader, self.support_loader, self.train_classes = self.data_handler.get_dataloader(
                    )
                    self.do_fs_train()
        else:
            self.data_loader = self.data_handler.get_dataloader()
            self.do_train()



    def do_train(self):
        """
        Training loop for base training.
        """
        self.logger = logging.getLogger("fcos_core.trainer")
        self.logger.info("Start training")
        self.meters = MetricLogger(delimiter="  ")
        self.max_iter = len(self.data_loader)
        start_iter = self.arguments["iteration"]
        self.model.train()
        start_training_time = time.time()
        end = time.time()

        len(self.data_loader)

        for iteration, (images, targets, _) in enumerate(self.data_loader, start_iter):
            data_time = time.time() - end
            iteration = iteration + 1
            self.arguments["iteration"] = iteration


            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            self.meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()
            self.meters.update(time=batch_time, data=data_time)

            eta_seconds = self.meters.time.global_avg * (self.max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == self.max_iter:
                self.logger.info(
                    self.meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(self.meters),
                        lr=self.optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                if sys.gettrace() is None:
                    self.tensorboard.add_multi_scalars(self.meters.to_dict(),
                                                   iteration)
            if iteration % self.logging_eval_int == 0:
                self.eval(iteration)

            if iteration % self.checkpoint_period == 0:
                self.checkpointer.save("model_{:07d}".format(iteration), **self.arguments)
            if iteration == self.max_iter:
                self.checkpointer.save("model_final", **self.arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (self.max_iter)
            )
        )

    def do_fs_train(self):
        """
        Training loop for few-shot training. Used for base training and finetuning. 
        """
        self.logger = logging.getLogger("fcos_core.trainer")
        self.logger.info("Start training")
        self.meters = MetricLogger(delimiter="  ")
        iter_epoch = len(self.query_loader)
        self.max_iter = iter_epoch * self.episodes + self.fintuning_start_iter

        if self.cfg.FINETUNE.ONLY or self.is_finetuning:
            start_iter = self.fintuning_start_iter
        else:
            start_iter = self.arguments["iteration"]
        self.model.train()
        start_training_time = time.time()
        end = time.time()

        self.data_handler.task_sampler.display_classes()

        for epoch in range(self.episodes):
            dataloader_seed = None if not self.is_finetuning else self.cfg.RANDOM.SEED
            self.query_loader, self.support_loader, self.train_classes = self.data_handler.get_dataloader(
                seed=dataloader_seed)
            print('Episode classes: {}'.format(str(self.train_classes)))
            for iteration, (images, targets, _) in enumerate(tqdm(self.query_loader), start_iter):

                # print(targets)
                data_time = time.time() - end
                iteration = epoch * iter_epoch + iteration + 1
                self.arguments["iteration"] = iteration

                # Main difference with do_train: support feature computation once per iteration
                support = self.model.compute_support_features(self.support_loader, self.device)

                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]

                loss_dict = self.model(images, targets, self.train_classes, support=support)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                self.meters.update(loss=losses_reduced, **loss_dict_reduced)

                self.optimizer.zero_grad()
                losses.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0) # use only for MFRCN to reduce unstability

                # print(self.model.support_features_extractor.body.layer4[2].conv3.
                #       weight.grad)

                self.optimizer.step()
                self.scheduler.step()

                batch_time = time.time() - end
                end = time.time()
                self.meters.update(time=batch_time, data=data_time)

                eta_seconds = self.meters.time.global_avg * (self.max_iter - iteration + start_iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                # Log both in console and tensorboard
                if iteration % self.logging_int == 0 or iteration == self.max_iter:
                    self.logger.info(
                        self.meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}",
                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(self.meters),
                            lr=self.optimizer.param_groups[0]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )
                    if sys.gettrace() is None:
                        self.tensorboard.add_multi_scalars(
                            self.meters.to_dict(), iteration)

                # Evaluation on validation set
                if iteration % self.logging_eval_int == 0:
                    self.eval_fs(iteration)

                # Model checkpointing
                if iteration % self.checkpoint_period == 0 or iteration == 1:
                    if self.is_finetuning:
                        model_name = "model_{:07d}_{}_shot".format(iteration, self.cfg.FEWSHOT.K_SHOT)
                    else:
                        model_name = "model_{:07d}".format(iteration)
                    self.checkpointer.save(model_name, **self.arguments)
                if iteration == self.max_iter:
                    model_name = "model_final" if not self.is_finetuning else "model_final_{}_shot".format(
                        self.cfg.FEWSHOT.K_SHOT)
                    self.checkpointer.save(model_name, **self.arguments)

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        self.logger.info(
            "Total {} time: {} ({:.4f} s / it)".format(
                'base training' if not self.is_finetuning else 'finetuning',
                total_time_str, total_training_time / (self.max_iter + 1)
            )
        )


    def eval_fs(self, iteration):
        """
        Evaluation during few-shot training.

        Create differenet Evaluator for base and novel classes, run eval and log metrics. 
        """
        self.model.eval()

        if self.evaluator_train is None or self.evaluator_test is None:
            self.init_eval()

        res_train, _ = self.evaluator_train.eval(verbose=False, all_classes=False, verbose_classes=False)
        res_test, _ = self.evaluator_test.eval(verbose=False, all_classes=False, verbose_classes=False)

        test_map, train_map = 0, 0
        if res_test != {}:
            test_map = res_test.stats[1]
        if res_train != {}:
            train_map = res_train.stats[1]

        eval_res = {'Train mAP': train_map, 'Test mAP': test_map}

        self.tensorboard.add_multi_scalars(eval_res, iteration, main_tag='Eval')
        self.model.train()

    def eval(self, iteration):
        """
        Regular evaluation (i.e. without few-shot).
        """
        self.model.eval()

        if self.evaluator_train is None or self.evaluator_test is None:
            data_handler = DataHandler(self.cfg,
                                       is_train=False,
                                       data_source='val')
            self.evaluator_train = Evaluator(self.model, self.cfg,
                                             data_handler)

        res_train, _ = self.evaluator_train.eval(verbose=False, all_classes=False, verbose_classes=False)

        train_map = 0

        if res_train != {}:
            train_map = res_train.stats[0]

        eval_res = {'Train mAP': train_map}

        self.tensorboard.add_multi_scalars(eval_res, iteration, main_tag='Eval')
        self.model.train()

    def init_eval(self):
        """
        Different DataHandler for base and novel classes as well as different 
        Evaluator. 
        """

        # Datahandler for validation dataset with train classes
        self.data_handler_val_train = DataHandler(self.cfg,
                                                  base_classes=True,
                                                  data_source='val',
                                                  is_train=False)

        # Datahandler for validation dataset with test classes
        self.data_handler_val_test = DataHandler(self.cfg,
                                                 base_classes=False,
                                                 data_source='val',
                                                 is_train=False)


        self.evaluator_train = Evaluator(self.model, self.cfg,
                                            self.data_handler_val_train)

        self.evaluator_test = Evaluator(self.model, self.cfg,
                                            self.data_handler_val_test)


    def prepare_finetuning(self, k_shot, episodes):
        self.episodes =episodes
        self.logging_int = self.cfg.LOGGING.INTERVAL // 1
        self.logging_eval_int = self.cfg.LOGGING.EVAL_INTERVAL // 3
        self.checkpoint_period = self.cfg.SOLVER.CHECKPOINT_PERIOD // 1

        self.checkpointer.load(os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth'))

        self.evaluator_train = None
        self.evaluator_test = None

        self.tensorboard = CustomLogger(
            log_dir='/home/pierre/Documents/PHD/Experiments/logs',
            notify=False)

        # Freeze backbone layer
        self.model.backbone.body._freeze_backbone(self.cfg.FINETUNE.FREEZE_AT)

        # Update dataloader
        self.data_handler = DataHandler(self.cfg,
                                        base_classes=not self.is_finetuning,
                                        is_train=True,
                                        start_iter=self.arguments['iteration'],
                                        is_finetune=self.is_finetuning)

        # Update optimizer (lr)
        del self.optimizer
        self.optimizer = make_optimizer(self.cfg, self.model, self.cfg.FINETUNE.LR)
        self.scheduler.milestones = [self.max_iter + s for s in self.cfg.FINETUNE.STEPS]

        # Update cfg
        self.cfg.merge_from_list(
            ['FEWSHOT.K_SHOT', k_shot, 
            'SOLVER.IMS_PER_BATCH', 4])
    

    def save_config(self):
        path_to_cfg_file = os.path.join(self.cfg.OUTPUT_DIR, 'model_cfg.yaml')
        with open(path_to_cfg_file, 'w') as f:
            f.write(self.cfg.dump())
