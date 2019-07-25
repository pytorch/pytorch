from ..dataset import IterableDataset  # noqa: F401
from torch._C.data.chunk import ChunkDatasetOptions  # noqa: F401


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

    def set_chunk_sampler_stride(self, stride):
        self.dataset.set_chunk_sampler_stride(stride)

    next = __next__  # py2 compatibility
