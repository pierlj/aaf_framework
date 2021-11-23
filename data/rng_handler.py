import random
import numpy as np
import torch

class RNGHandler():
    """ 
    Manage rng for random, numpy and pytorch as they all use
    different seeds. 

    Plus additional functions for easy control about pytorch determinism. 
    """
    def __init__(self, cfg):
        self.seed = cfg.RANDOM.SEED
        self.determinism = cfg.RANDOM.DISABLED
        self.set_seeds()
        self.np_rng = np.random.RandomState(self.seed)
        self.rn_rng = random.Random(self.seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(self.seed)

    def set_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.determinism:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # torch.use_deterministic_algorithms(True)

    def update_seeds(self, seed):
        self.np_rng = np.random.RandomState(seed)
        self.rn_rng = random.Random(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)

    def enable_determinism(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # torch.use_deterministic_algorithms(False)


    def worker_init_fn(self):
        def f(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        return f
