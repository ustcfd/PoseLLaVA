from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

class BaseDatasetBuilder(ABC, object):
    def __init__(self, random_seed = None) -> None:
        self.random_seed = random_seed

    @abstractmethod
    def __call__(self, data_args, ) -> Dataset: ...

class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size