import math
import torch
from torch.utils.data import Sampler

class FilteringSampler(Sampler):
    """
    Wraps a RandomSampler, sample a specific number
    of image to build a query set, i.e. a set that 
    contains a fixed number of images of each selected
    class.
    """

    def __init__(self, dataset, selected_classes, n_query, shuffle, rng):
        self.dataset = dataset

        self.selected_classes = selected_classes
        self.n_query = n_query
        self.shuffle = shuffle
        self.rng = rng

    def __iter__(self):
        table = self.dataset.class_table
        selected_indices = []
        for c in self.selected_classes:
            class_id = int(c.item())
            keep = torch.randperm(len(table[class_id]), generator=self.rng)[:self.n_query]
            selected_indices = selected_indices + [
                self.dataset.image_ids_to_ids[table[class_id][k]] for k in keep]
        selected_indices = torch.Tensor(selected_indices)

        # Retrieve indices inside dataset from img ids
        if self.shuffle:
            shuffle = torch.randperm(selected_indices.shape[0], generator=self.rng)
            yield from selected_indices[shuffle].long().tolist()
        else:
            yield from selected_indices.long().tolist()

    def __len__(self):
        length = 0
        for c in self.selected_classes:
            cls = int(c.item())
            table = self.dataset.class_table
            if self.n_query > 0:
                length += min(
                    self.n_query,
                    len(table[cls]))
            else:
                length += len(table[cls])

        return length

class SupportSampler(Sampler):
    """
    Wraps a RandomSampler, sample a specific number
    of image to build a query set, i.e. a set that 
    contains a fixed number of images of each selected
    class.
    """

    def __init__(self,
                 dataset,
                 selected_examples):
        self.dataset = dataset
        self.selected_examples = selected_examples

    def __iter__(self):
        table = self.dataset.class_table
        selected_indices = []

        for c, keep in self.selected_examples.items():
            selected_indices = selected_indices + [
                (self.dataset.image_ids_to_ids[table[c][k]], int(c))
                for k in keep
            ]

        # Retrieve indices inside dataset from img ids
        # selected_indices = torch.Tensor(selected_indices).long()
        # shuffle = torch.randperm(selected_indices.shape[0])

        yield from selected_indices

    def __len__(self):
        length = 0
        for _, examples in self.selected_examples.items():
            length += len(examples)

        return length


# class FinetuningSampler(Sampler):
#     """
#     Wraps a RandomSampler, sample a specific number
#     of image to build a query set, i.e. a set that 
#     contains a fixed number of images of each selected
#     class.
#     """
#     def __init__(self,
#                  dataset,
#                  selected_examples):
#         # during finetuning the seed of this rng handler is reset at each episode
#         # so that it samples always the same novel examples from dataset

#         self.dataset = dataset
#         self.selected_examples = selected_examples


#     def __iter__(self):
#         table = self.dataset.class_table
#         selected_indices = []

#         for c, keep in self.selected_examples.items():
#             selected_indices = selected_indices + [
#                 (self.dataset.image_ids_to_ids[table[c][k]], int(c)) for k in keep
#             ]

#         # Retrieve indices inside dataset from img ids
#         # selected_indices = torch.Tensor(selected_indices).long()
#         # shuffle = torch.randperm(selected_indices.shape[0])

#         yield from selected_indices

#     def __len__(self):
#         length = 0
#         for _, examples in self.selected_examples.items():
#             length += len(examples)

#         return length

#     def select_classes(self, train, test):
#         nb_classes = len(train)
#         n_train, n_test = math.ceil(nb_classes / 2), math.floor(nb_classes /
#                                                                  2)
#         rng = self.rng_free.rn_rng
#         train, test = train.copy(), test.copy()
#         rng.shuffle(train)
#         rng.shuffle(test)
#         selected_classes = train[:n_train] + test[:n_test]
#         print(selected_classes)
#         return sorted(selected_classes)
