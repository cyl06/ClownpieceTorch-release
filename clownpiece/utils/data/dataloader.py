from clownpiece.utils.data.dataset import Dataset
from clownpiece import Tensor
from utils_ import *
import numpy as np

class DefaultSampler:
    def __init__(self, length, shuffle):
        self.length = length
        self.shuffle = shuffle

    def __iter__(self):
        self.indices = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return self.length

def default_collate_fn(batch):
    if not batch:
        return batch
    if not isinstance(batch[0], (tuple, list)):
        if isinstance(batch[0], Tensor):
            return Tensor.stack([item for item in batch])
        if isinstance(batch[0], (int, float)):
            return Tensor(list(batch))
        raise TypeError(f"IDK")
    trans = list(zip(*batch))
    return tuple(default_collate_fn(sample) for sample in trans)

class Dataloader:
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 drop_last=False, 
                 sampler=None, 
                 collate_fn=None):
        self.dataset = dataset
        self.batch_siz = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        if sampler:
            self.sampler = sampler
        else:
            self.sampler = DefaultSampler(len(dataset), shuffle)
        if collate_fn:
            self.collate_fn = collate_fn
        else:
            self.collate_fn = default_collate_fn

    def __iter__(self):
        # yield a batch of data
        batch = []
        for idx in self.sampler:
            batch.append(self.dataset[idx])
            if len(batch) == self.batch_siz:
                yield self.collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self):
        # number of batches, not the number of items in dataset
        if self.drop_last:
            return len(self.dataset) // self.batch_siz
        else:
            return ceil_div(len(self.dataset), self.batch_siz)