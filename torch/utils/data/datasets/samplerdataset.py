from torch.utils.data import IterableDataset, Sampler, SequentialSampler
from typing import TypeVar, Iterator, Sized

T_co = TypeVar('T_co', covariant=True)


class SamplerIterableDataset(IterableDataset[T_co]):
    r""" Prototype of :class:`SamplerIterableDataset`.

    IterableDataset to generate samples elements.
    args:
        dataset: IterableDataset being collated
        sampler: Sampler to genereate sample elements from input dataset.
                    Default is :class:`SequentialSampler` for IterableDataset
    """
    def __init__(self,
                 dataset: IterableDataset[T_co],
                 *,
                 sampler: Sampler = SequentialSampler,
                 ) -> None:
        self.dataset = dataset
        self.sampler = sampler(self.dataset)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        if isinstance(self.dataset, Sized) and \
           isinstance(self.sampler, Sized) and \
           len(self.sampler) >= 0:
            return len(self.sampler)
        else:
            raise NotImplementedError
