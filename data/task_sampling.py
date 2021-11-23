import os
import math
import torch

from fcos_core.utils.imports import import_file
class TaskSampler():
    """
    Class used for managing classes splits during training: 
        - Base/novel split
        - Episode class sampling
    """
    def __init__(self, cfg, classes, rng, eval=False, is_finetune=False):
        """
        Parameters: 
            - classes: list of all classes
            - cfg: cfg object for the current training 
        """

        self.classes = classes
        self.cfg = cfg
        self.rng = rng
        self.eval = eval
        self.is_finetune = is_finetune

        self.split_train_test_classes(cfg.FEWSHOT.SPLIT_METHOD)

    def split_train_test_classes(self, split_method='random'):
        if split_method =='random':
            self.c_test = self.rng.sample(self.classes,
                                     k=self.cfg.FEWSHOT.N_CLASSES_TEST)
            self.c_test.sort()
            self.c_train = [c for c in self.classes if c not in self.c_test]

        elif split_method == 'same':
            assert self.cfg.FEWSHOT.N_WAYS_TEST == self.cfg.FEWSHOT.N_WAYS_TRAIN, 'N_WAYS_TRAIN and N_WAYS_TEST should be equal when using "same" samling'
            self.c_test = self.rng.sample(self.classes,
                                     k=self.cfg.FEWSHOT.N_CLASSES_TEST)
            self.c_test.sort()
            self.c_train = self.c_test

        elif split_method == 'deterministic':
            paths_catalog = import_file(
                "fcos_core.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            DatasetCatalog = paths_catalog.DatasetCatalog
            self.dataset_path = '/'.join(getattr(DatasetCatalog,
                    'DATASETS')[self.cfg.DATASETS.TRAIN[0]]['img_dir'].split('/')[:-1])
            with open(os.path.join(self.dataset_path, 'classes_split.txt'), 'r') as f:
                lines = f.readlines()
                assert len(lines)== 2, 'Wrong classes split file format, should be: \ntrain_classes:1,2,3 \n test_classes:4,5,6'
                self.c_train = list(map(lambda x: int(x), lines[0][:-1].split(':')[-1].split(',')))
                self.c_test = list(map(lambda x: int(x), lines[1][:-1].split(':')[-1].split(',')))
                assert len(self.c_train) >= self.cfg.FEWSHOT.N_WAYS_TRAIN, 'N_WAYS_TRAIN too large for number of training classes defined in file'
                assert len(self.c_test) >= self.cfg.FEWSHOT.N_WAYS_TEST, 'N_WAYS_TEST too large for number of training classes defined in file'

                self.c_train.sort()
                self.c_test.sort()

    def sample_train_val_tasks(self, n_ways_train, n_ways_test, verbose=False):
        """
        Create two tasks for the episode (train and test) by sampling classes within the allowed range.

        Parameters:
        - n_ways_train: number of classes for training task
        - n_ways_test: number of classes for testing task
        - verbose: outputs information about classes selected for the current tasks

        Returns:
        - train_task: Query set, Support set, task_classes
        - test_task: Query set, Support set, task_classes
        """
        if self.eval:
            # replace by batch tasks
            train_task = self.batch_tasks(n_ways_train,
                                          self.c_train)
            test_task = self.batch_tasks(n_ways_test,
                                         self.c_test)
            return train_task, test_task
        else:
            train_task = self.sample_task(
                n_ways_train, self.c_train, verbose=verbose)
            test_task = self.sample_task(
                n_ways_test, self.c_test, verbose=verbose)
            return train_task, test_task

    def sample_task(self, n_ways, classes_from, verbose=False):
        """
        Sample classes for a task and create dataset objects for support and query sets. 
        
        Parameters:
        - n_ways: number of classes for the task
        - classes_from: classes list from which classes are chosen
        - verbose: display selected classes for current task
        """
        assert n_ways <= len(classes_from), 'Not enough classes for this task, either n_ways is too big or classes_from is too small'

        self.c_episode = self.rng.sample(classes_from, k=n_ways)

        self.c_episode.sort()
        if verbose:
            print('Selected classes: {}'.format(self.c_episode))


        return self.c_episode

    def batch_tasks(self, n_ways, classes_from):
        """
        Create batches of classes. 
        
        Parameters:
        - n_ways: number of classes for the task
        - classes_from: classes list from which classes are chosen
        - verbose: display selected classes for current task
        """
        assert n_ways <= len(
            classes_from
        ), 'Not enough classes for this task, either n_ways is too big or classes_from is too small'

        classes = classes_from.copy()
        self.rng.shuffle(classes)

        n_batch = math.ceil(len(classes) / n_ways)
        batches = []
        for i in range(n_batch):
            batch = classes[i * n_ways:(i + 1) * n_ways]
            if len(batch) < n_ways:
                batch = batch + classes[:(n_ways-len(batch))]
            batch.sort()
            batches.append(batch)

        return batches

    def display_classes(self):
        train_set_classes = ', '.join([str(c) + " " + str(c) for c in self.c_train])
        val_set_classes = ', '.join([str(c) + " " + str(c)
                             for c in self.c_test])
        print("""Selected categories:
                Train support set: {}
                Validation support set: {}""".format(train_set_classes, val_set_classes))
    
    def finetuning_classes_selection(self, train_classes, test_classes, rng_handler):
        """
        Selects randomly classes for finetuning by mixing base and novel classes.
        """
        train_classes, test_classes = torch.Tensor(train_classes), torch.Tensor(test_classes)
        nb_classes = len(train_classes)
        n_train, n_test = math.ceil(nb_classes / 2), math.floor(nb_classes /
                                                                 2)
        rng = rng_handler.torch_rng
        keep_train = torch.randperm(nb_classes, generator=rng)[:n_train]
        keep_test = torch.randperm(nb_classes, generator=rng)[:n_test]

        selected_classes = torch.cat([train_classes[keep_train], test_classes[keep_test]])
        return selected_classes.sort()[0].tolist()
