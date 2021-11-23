import os
import torch

from ..utils.utils import random_choice
from ..utils import find_id_in_coco_img

class ExampleSelector():
    """
    Class that contains tools to sampled examples from the datasets. 
    """
    def __init__(self, cfg, is_train, is_finetune, rng_handler):

        self.cfg = cfg
        self.is_train = is_train
        self.is_finetune = is_finetune

        self.selected_novel_examples = {}
        self.selected_base_examples = {}

        self.rng_handler = rng_handler



    def select_examples(self, dataset, selected_classes, n_query, test_classes):
        """
        Select elements inside dataset for both query and support set.
        Different behaviors are expected when finetuning is on or off 
        and following the network params.

        Parameters:
            dataset: dataset object in which sampling is allowed
            selected_classes: classes selected for the sampling
            n_query: number of example for each class
        
        Returns:


        """
        self.test_classes = test_classes

        table = dataset.class_table
        # finetuning examples are selected randomly at each episode
        if self.is_finetune:
            if self.cfg.FINETUNE.EXAMPLES =='random':
                self.selected_base_examples = self.select_random_idx_per_class(selected_classes, table, n_query)

            # finetuning examples for novel classes are randomly sampled
            # once and re-use at each episode. Other classes examples
            # are still sampled randomly
            elif self.cfg.FINETUNE.EXAMPLES =='rng_same':
                if self.selected_novel_examples == {}:
                    self.selected_novel_examples = self.select_random_idx_per_class(self.test_classes, table, n_query)
                self.selected_base_examples = self.select_random_idx_per_class(selected_classes, table, n_query)
                self.selected_base_examples.update({k:v for k,v in self.selected_novel_examples.items() if float(k) in selected_classes})

            # finetuning examples for novel classes are selected deterministically
            # from file as in Kang et al (see https://github.com/bingykang/Fewshot_Detection/tree/master/data).
            # Other classes examples are still sampled randomly
            elif self.cfg.FINETUNE.EXAMPLES =='deterministic':
                if self.selected_novel_examples == {}:
                    test_classes = self.test_classes
                    if type(self.test_classes[0]) == list:
                        test_classes = [c for c_ in self.test_classes for c in c_]

                    self.selected_novel_examples = self.select_deterministic_idx(
                                                test_classes, n_query, dataset)

                self.selected_base_examples = self.select_random_idx_per_class(
                                                selected_classes, table, n_query)
                self.selected_base_examples.update({
                    k: v
                    for k, v in self.selected_novel_examples.items()
                    if float(k) in selected_classes
                })
            else:
                raise NotImplementedError
        else:

            if not self.is_train and self.cfg.FINETUNE.EXAMPLES =='deterministic':
                if self.selected_novel_examples == {}:
                    test_classes = self.test_classes
                    if type(self.test_classes[0]) == list:
                        test_classes = [
                            c for c_ in self.test_classes for c in c_
                        ]

                    self.selected_novel_examples = self.select_deterministic_idx(
                                                test_classes, n_query, dataset)

                self.selected_base_examples = self.select_random_idx_per_class(
                                                selected_classes, table, n_query)
                self.selected_base_examples.update({
                    k: v
                    for k, v in self.selected_novel_examples.items()
                    if float(k) in selected_classes
                })
            else:
                self.selected_base_examples = self.select_random_idx_per_class(
                    selected_classes, table, n_query)

        return self.selected_base_examples, self.selected_novel_examples

    def select_random_idx_per_class(self, classes, table, n_query):
        """
        Randomly picks indices from class table. 
        """
        selected_idx = {}
        for c in classes:
            class_id = int(c)

            keep = random_choice(len(table[class_id]),
                                 n_query,
                                 generator=self.rng_handler.torch_rng)
            selected_idx[class_id] = keep
        return selected_idx

    def select_deterministic_idx(self, classes, n_query, dataset):
        """
        Find indices of the images in dataset according to the split files.

        Note that the path and file name could differ.

        TO DO: add flexibility for path and file name, as a class attribute
        of ExampleSelector.
        """

        # n_query must be the number of shot here
        split_path = os.path.join('/'.join(dataset.root.split('/')[:-1]), 'annotations', 'splits')
        split_master_file = 'voc_traindict_bbox_{}shot.txt'.format(n_query)

        with open(os.path.join(split_path, split_master_file), 'r') as f:
            lines = f.readlines()

        classes_cat = {dataset.coco.cats[
                        dataset.contiguous_category_id_to_json_id[c]]['name'] : int(c)
                        for c in classes
        }

        img_paths = {}
        for l in lines:
            c, split_file_path = l.split(' ')
            if c in classes_cat:
                with open(split_file_path[:-1], 'r') as f:
                    img_paths[classes_cat[c]] = f.readlines()

        selected_idx = {}
        for class_id, paths_to_ex in img_paths.items():
            keep = []
            for ex in paths_to_ex:
                img_name = ex.split('/')[-1][:-1]
                img_id = find_id_in_coco_img(
                    dataset.coco, 'file_name', img_name)
                id_in_class_table = dataset.class_table[class_id].index(img_id)
                keep.append(torch.tensor([id_in_class_table]))
            selected_idx[class_id] = torch.cat(keep)
            if len(selected_idx[class_id]) < n_query:
                selected_idx[class_id] = torch.cat([
                    selected_idx[class_id],
                    self.select_random_idx_per_class(
                        [int(class_id)], dataset.class_table,
                        n_query - len(selected_idx[class_id]))[class_id]
                ])

        return selected_idx