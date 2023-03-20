from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, DistributedSampler, BatchSampler
import os
import torch
from rvlcdip import RvlDataset
from samplers import get_imbalanced_sampler, DistributedSamplerWrapper
from torch import distributed as dist
# from catalyst.data import DynamicBalanceClassSampler


class DataPipeline(LightningDataModule):
    def __init__(self, root, batch_size=16, distributed=False) -> None:
        super(DataPipeline, self).__init__()
        self.root = root
        self.batch_size = batch_size
        self.distributed = distributed

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                root=self.root,
                split="train",
            )

            # train_size = int(0.99 * len(self.train_dataset))
            # test_size = len(self.train_dataset) - train_size
            # _, self.train_dataset = torch.utils.data.random_split(
            #     self.train_dataset, [train_size, test_size])
            # self.val_dataset = self.train_dataset
            self.val_dataset = DataPipeline.get_dataset(
                root=self.root,
                split="val",
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                root=self.root,
                split="test",
            )

    def train_dataloader(self) -> DataLoader:
        # targets = self.train_dataset.targets
        # sampler = DynamicBalanceClassSampler(targets)

        # sampler = get_imbalanced_sampler(targets, replacement=True)
        # if self.distributed:
        #     if dist.is_available() and dist.is_initialized():
        #         local_rank = dist.get_rank()
        #         world_size = dist.get_world_size()

        #         sampler = DistributedSamplerWrapper(
        #             sampler,
        #             num_replicas=world_size,
        #             rank=local_rank,
        #             shuffle=False
        #         )

        # dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
        #                         sampler=sampler, shuffle=False, num_workers=10, drop_last=True)
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=16, drop_last=True)

        return dataloader

    def val_dataloader(self) -> DataLoader:
        if self.distributed:
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(self.val_dataset)

        data_loader_val = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            # sampler=sampler,
            drop_last=False, num_workers=16,
        )
        return data_loader_val

    def test_dataloader(self) -> DataLoader:
        if self.distributed:
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(self.test_dataset)

        data_loader_test = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # sampler=sampler,
            drop_last=False, num_workers=16,
        )
        return data_loader_test

    @classmethod
    def get_dataset(cls, root, split) -> Dataset:
        tar_path = os.path.join(root, split+'.tar')
        dataset = RvlDataset(tar_path=tar_path)
        return dataset

    def get_samples_per_class(self):
        labels = self.train_dataset.targets
        return torch.bincount(labels).tolist()
