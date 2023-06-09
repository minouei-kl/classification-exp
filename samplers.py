from operator import itemgetter
from typing import Optional

import torch
from torch.utils.data import DistributedSampler, Dataset
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
import numpy as np


def get_imbalanced_sampler(labels,  replacement=True):

    num_samples = len(labels)
    labels = torch.LongTensor(np.array(labels))
    class_count = torch.bincount(labels).to(dtype=torch.float)
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=num_samples, replacement=replacement)
    return sampler


# https://github.com/catalyst-team/catalyst/blob/5f3a2a87f2179f336fe6dab5fc88359cf0b7e86d/catalyst/data/dataset/torch.py#L203-L232
class DatasetFromSampler(Dataset):
    """Dataset of indexes from `Sampler`."""

    def __init__(self, sampler: Sampler):
        """
        Args:
            sampler (Sampler): @TODO: Docs. Contribution is welcome
        """
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


# borrowed from: https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
            self,
            sampler,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


# if __name__ == "__main__":
    # if dist.is_available() and dist.is_initialized():
    #     local_rank = dist.get_rank()
    #     world_size = dist.get_world_size()

    # imbalanced_sampler = get_imbalanced_sampler(self.train_df.landmark_id, num_samples=n_samples,
    #                                             replacement=self.replacement)
    # distributed_sampler = DistributedSamplerWrapper(
    #     imbalanced_sampler,
    #     num_replicas=world_size,
    #     rank=local_rank,
    #     shuffle=False
    # )
