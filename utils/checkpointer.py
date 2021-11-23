# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from fcos_core.utils.model_serialization import load_state_dict
from fcos_core.utils.c2_model_loading import load_c2_format
from fcos_core.utils.imports import import_file
from fcos_core.utils.model_zoo import cache_url

from .utils import DisableLogger


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = "{}.pth".format(name)
        save_path = os.path.join(self.save_dir, save_file)

        self.logger.info("Saving checkpoint to {}".format(save_path))
        torch.save(data, save_path)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info(
                "No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)

        if 'model_final_checkpoint_phase499' in f:
            self._load_ssl_model(checkpoint)
        else:
            self._load_model(checkpoint)

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        with DisableLogger():
            load_state_dict(self.model, checkpoint.pop("model"))

    def _load_ssl_model(self, checkpoint):
        model_state_dict = self.model.state_dict()
        # load_state_dict(self.model, checkpoint.pop("classy_state_dict"))
        trunk_weights = checkpoint['model']["classy_state_dict"]["base_model"]["model"]["trunk"]
        prefix = "_feature_blocks."

        # Remove prefix in front of layer block, add it for layer0 or
        # stem layer with FCOS convention and discard completely
        # num_batches_tracked from BN layers as won't be used in FrozenBN
        trunk_weights = {k[len(prefix):] : w for k, w in trunk_weights.items()}
        trunk_weights = {(k if 'layer' in k else 'stem.'+k ):w for k, w in trunk_weights.items()}
        trunk_weights = {k:w for k, w in trunk_weights.items() if 'num_batches_tracked' not in k}

        for k, w in trunk_weights.items():
            model_state_dict['backbone.body.' + k] = w
        
        self.model.load_state_dict(model_state_dict)


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
        testing=False,
    ):
        super(DetectronCheckpointer,
              self).__init__(model, optimizer, scheduler, save_dir,
                             save_to_disk, logger)
        self.cfg = cfg.clone()
        self.testing = testing

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file("fcos_core.config.paths_catalog",
                                        self.cfg.PATHS_CATALOG, True)
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f

        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            with DisableLogger():
                return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def has_checkpoint(self):
        return (self.testing or self.cfg.SOLVER.CONTINUE_TRAINING) \
                 and super(DetectronCheckpointer, self).has_checkpoint()
