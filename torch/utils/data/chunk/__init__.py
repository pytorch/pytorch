from ..dataset import IterableDataset
from torch._C.data.chunk import ChunkDatasetOptions, SamplerWrapper
from torch._C.data.chunk import Sampler, RandomSampler, SequentialSampler
from torch._C.data.chunk import DistributedSampler, DistributedRandomSampler, DistributedSequentialSampler
from torch._C.data.chunk import ChunkDataReaderUint8T, ChunkDataReaderInt8T, ChunkDataReaderInt16T, ChunkDataReaderInt32T
from torch._C.data.chunk import ChunkDataReaderInt64T, ChunkDataReaderFloat, ChunkDataReaderDouble

class ChunkDatasetWrapper(IterableDataset):
    r"""
    Wrapper class for the C++ ChunkDataset class.

    ``ChunkDatasetWrapper`` extends ``IterableDataset`` because the 
    size of the ChunkDataset doesn't know the size of the dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
    """
    def __init__(self, dataset):
        super(ChunkDatasetWrapper, self).__init__()
        self.dataset = dataset

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError()

    def __next__(self):
        batch = self.dataset.get_batch()
        if batch is None:
            raise StopIteration
        return batch

    def reset(self):
        self.dataset.reset()

    def chunk_sampler(self):
        return self.dataset.chunk_sampler()

    next = __next__  # py2 compatibility