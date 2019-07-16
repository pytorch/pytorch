from ..dataset import IterableDataset
from torch._C.data.chunk import ChunkDatasetOptions, SamplerWrapper
from torch._C.data.chunk import Sampler, RandomSampler, SequentialSampler
from torch._C.data.chunk import DistributedSampler, DistributedRandomSampler, DistributedSequentialSampler
from torch._C.data.chunk import ChunkDataReaderUint8T, ChunkDataReaderInt8T, ChunkDataReaderInt16T, ChunkDataReaderInt32T
from torch._C.data.chunk import ChunkDataReaderInt64T, ChunkDataReaderFloat, ChunkDataReaderDouble

class ChunkDataset(IterableDataset):
    r"""
    Wrapper class for the C++ ChunkDataset class.

    Additional to the role of wrapping its C++ counterpart dataset class,
    it also allows collation, convertion or transformation through ``collate_fn``.
    ``ChunkDataset`` extends ``IterableDataset`` because the size of the dataset is
    unknown.

    Arguments:
        dataset (Dataset): The whole Dataset
        collate_fn (optional, callable): Collates, converts or transform batches
    """
    def __init__(self, dataset, collate_fn=None):
        super(ChunkDataset, self).__init__()
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError()

    def __next__(self):
        batch = self.dataset.get_batch()
        if batch is None:
            raise StopIteration
        if self.collate_fn is not None:
            return self.collate_fn(batch)
        return batch

    def reset(self):
        self.dataset.reset()

    def chunk_sampler(self):
        return self.dataset.chunk_sampler()

    next = __next__  # py2 compatibility