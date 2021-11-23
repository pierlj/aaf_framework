import math
import os

import torch
from fcos_core.utils.imports import import_file
from torch.utils.data.sampler import BatchSampler


from . import datasets as D
from . import samplers
from .build import build_dataset, make_batch_data_sampler
from .collate_batch import BatchCollator, BBoxAugCollator
from .example_selector import ExampleSelector
from .rng_handler import RNGHandler
from .samplers.fs_sampler import FilteringSampler, SupportSampler
from .task_sampling import TaskSampler
from .transforms import build_transforms
from ..utils import HiddenPrints


class DataHandler():
    def __init__(self,
                cfg,
                base_classes=True,
                start_iter=0,
                data_source='train',
                eval_all=False,
                is_train=False,
                is_finetune=False):
        """
            Object to manage datasets and dataloader. It can be used in a fewshot setting
            yielding loader for query and support sets or in a regular manner (only one loader).
            
            When eval_all is true sample batch of tasks to eval performances on all classes.
            - cfg: Config object for the whole network
            - base_classes: controls wether data should contains annotations either from base or novel classes
            - start_iter: start at iteration start_iter when restarting training
            - data_source: either 'train', 'val' or 'test' controls from which dataset split data is loaded
            - eval_all: special mode that samples test episodes and respective loader for evaluating on all classes (base and novel)
            - is_train: specifies whether network is training or not to select parameters accordingly
            - is_finetune: specifies whether network is finetuning or not. This changes how examples are selected in dataset
        """

        self.cfg = cfg
        self.base_classes = base_classes
        self.start_iter = start_iter
        self.data_source = data_source
        self.eval_all = eval_all
        self.is_train = is_train
        self.is_finetune = is_finetune

        # To properly manage sampling for both base and novel classes two
        # rng handler are used. rng_handler_fixed's seed is fixed before the sampling of
        # the dataloader to fix it.
        self.rng_handler_fixed = RNGHandler(cfg)
        self.rng_handler_free = RNGHandler(cfg)

        self.categories = None
        self.selected_base_examples = {} #
        self.selected_novel_examples = {}

        self.example_selector = ExampleSelector(cfg, is_train, is_finetune, self.rng_handler_free)

        with HiddenPrints():
            self.build_datasets()
        # self.build_datasets()

        self.n_classes = len(self.datasets[0].coco.cats)
        self.task_sampler = TaskSampler(cfg,
                                        [i + 1 for i in range(self.n_classes)],
                                        rng=self.rng_handler_fixed.rn_rng,
                                        eval=eval_all)


    def get_dataloader(self, seed=None):
        """
        Return either 1 dataloader for the whole training, with MAX_ITER 
        iteration. This is done when FSL is disabled.

        Or 2 dataloaders, 1 for query set and 1 for support set.

        """
        if self.cfg.FEWSHOT.ENABLED:

            # Task sampling
            self.train_classes, self.test_classes = self.task_sampler.sample_train_val_tasks(
                                                        self.cfg.FEWSHOT.N_WAYS_TRAIN,
                                                        self.cfg.FEWSHOT.N_WAYS_TEST,
                                                        verbose=False
                                                    )

            if self.eval_all:
                """
                When eval_all is true, query and support loaders are created
                for each batch of classes output from the task_sampler. 
                These are divided into train and test classes. 
                """
                loaders = {'train': [],
                            'test': []}
                for test_cls in self.test_classes:
                    loaders['test'].append(
                        self.get_two_loaders(
                            test_cls, [self.datasets, self.support_datasets], seed))
                for train_cls in self.train_classes:
                    loaders['train'].append(
                        self.get_two_loaders(
                            train_cls, [self.datasets, self.support_datasets], seed))

                return loaders

            elif self.is_finetune and self.cfg.FINETUNE.MIXED:
                classes = self.task_sampler.finetuning_classes_selection(
                    self.train_classes, self.test_classes, self.rng_handler_free)
            else:
                classes = self.train_classes if self.base_classes else self.test_classes

            return self.get_two_loaders(
                classes, [self.datasets, self.support_datasets], seed)
        else:
            selected_classes = torch.Tensor(
                [i + 1 for i in range(self.n_classes)])

            return self.make_data_loader_filtered(selected_classes, self.datasets, is_fewshot=False)

    def get_two_loaders(self, classes, datasets, seed=None):
        """
        Arguments:
            classes: list of classes
            datasets: list of datasets list first element is query dataset, second is support

        Return two dataloaders: one for query set and one for support and the set of classes. 
        """
        if seed is not None:
            self.rng_handler_fixed.update_seeds(seed)

        return (self.make_data_loader_filtered(torch.Tensor(classes),
                                            datasets[0], seed=seed),
                self.make_data_loader_filtered(torch.Tensor(classes),
                                            datasets[1],
                                            is_support=True, seed=seed),
                classes)


    def make_data_loader_filtered(self, selected_classes, datasets, is_support=False, is_fewshot=True, seed=None):
        """
            Select parameters for the creation of the dataloader

            Arguments:
                selected_classes: list of classes that should be annotated in the loader
                datasets: list of dataset in which load the data
                is_support: controls whether it a support or query loader
                is_fewshot: controls whether loader should be for fewshot use or not
            
            Returns:
        """

        shuffle, num_iters, start_iter, aspect_grouping, images_per_gpu = self.get_parameters(is_fewshot)

        data_loaders = []
        for dataset in datasets:
            # will work only for cocodataset

            dataset.selected_classes = selected_classes
            dataset.is_support = is_support

            if is_fewshot:
                if is_support:
                    n_query = self.cfg.FEWSHOT.K_SHOT
                    if not self.cfg.FEWSHOT.SAME_SUPPORT_IN_BATCH:
                        # when same support in batch is deactivated
                        # one support should be sampled for each
                        # element of the batch
                        n_query = n_query * images_per_gpu

                    self.select_examples(dataset, selected_classes, n_query)
                    sampler = SupportSampler(dataset, self.selected_base_examples)
                    batch_sampler = BatchSampler(sampler, images_per_gpu, drop_last=False)
                else:
                    if self.is_finetune:
                        self.select_examples(dataset, selected_classes,
                                             self.cfg.FEWSHOT.K_SHOT)
                        sampler = SupportSampler(
                            dataset,
                            self.selected_base_examples)
                    else:
                        n_query = self.cfg.FEWSHOT.N_QUERY_TRAIN if self.is_train else \
                            self.cfg.FEWSHOT.N_QUERY_TEST
                        sampler = FilteringSampler(dataset, selected_classes, n_query, shuffle,
                            rng=self.rng_handler_fixed.torch_rng)
                    batch_sampler = BatchSampler(sampler,
                                                 images_per_gpu,
                                                 drop_last=False)
            else:
                # Large n_query value will gather all element within each class in the dataset
                sampler = FilteringSampler(dataset, selected_classes, len(dataset), shuffle,
                    rng=self.rng_handler_fixed.torch_rng)
                batch_sampler = make_batch_data_sampler(dataset,
                                                        sampler,
                                                        aspect_grouping,
                                                        images_per_gpu,
                                                        num_iters,
                                                        start_iter,
                                                        is_support=is_support)


            collator = BBoxAugCollator() if not self.is_train and self.cfg.TEST.BBOX_AUG.ENABLED else \
                BatchCollator(self.cfg.DATALOADER.SIZE_DIVISIBILITY)
            num_workers = self.cfg.DATALOADER.NUM_WORKERS
            data_loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collator,
                worker_init_fn=self.rng_handler_fixed.worker_init_fn(),
                generator=self.rng_handler_fixed.torch_rng
            )
            data_loaders.append(data_loader)

        self.build_categories_map(data_loaders[0].dataset)
        return data_loaders[0]

    def build_datasets(self):
        """
        Select right parameters and dataset class and create datasets
        both for query and support (when FS is enabled).
        """

        paths_catalog = import_file(
            "fcos_core.config.paths_catalog", self.cfg.PATHS_CATALOG, True
        )
        DatasetCatalog = paths_catalog.DatasetCatalog
        dataset_list = getattr(self.cfg.DATASETS, self.data_source.upper())

        # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
        transforms = None if self.is_train and self.cfg.TEST.BBOX_AUG.ENABLED else build_transforms(self.cfg, self.is_train)
        self.datasets = build_dataset(
            dataset_list,
            transforms,
            DatasetCatalog,
            is_train=self.is_train,
            cfg=self.cfg,
            mode='finetune' if self.is_finetune else 'train')

        if self.cfg.FEWSHOT.ENABLED:
            if not self.is_train and self.cfg.FINETUNE.EXAMPLES == 'deterministic':
                dataset_list = getattr(self.cfg.DATASETS, 'TRAIN')

            self.support_datasets = build_dataset(dataset_list,
                                                  transforms,
                                                  DatasetCatalog,
                                                  cfg=self.cfg,
                                                  is_train=self.is_train,
                                                  mode='support')

    def get_parameters(self, is_fewshot):
        num_gpus = 1
        if self.is_train:
            images_per_batch = self.cfg.SOLVER.IMS_PER_BATCH
            images_per_gpu = images_per_batch // num_gpus
            shuffle = True

            if is_fewshot:
                num_iters = None
            else:
                num_iters = self.cfg.SOLVER.MAX_ITER # Important for not repeating same examples over and over

            start_iter = self.start_iter
        else:
            images_per_batch = self.cfg.TEST.IMS_PER_BATCH
            images_per_gpu = images_per_batch // num_gpus
            shuffle = True
            num_iters = None
            start_iter = 0

        aspect_grouping = [1] if self.cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

        return shuffle, num_iters, start_iter, aspect_grouping, images_per_gpu

    def build_categories_map(self, coco_dataset):
        cats = coco_dataset.coco.cats
        cat_id_map = coco_dataset.json_category_id_to_contiguous_id
        self.categories = {}
        for k, v in cats.items():
            self.categories[cat_id_map[k]] = v['name']

    def select_examples(self, dataset, selected_classes, n_query):
        self.selected_base_examples, self.selected_novel_examples = \
            self.example_selector.select_examples(dataset,
                                                selected_classes,
                                                n_query, self.test_classes)
