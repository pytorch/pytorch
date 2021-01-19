from torch.utils.data import IterableDataset, Sampler, SequentialSampler
from typing import TypeVar, Type, Iterator, Sized

T_co = TypeVar('T_co', covariant=True)


class SamplerIterableDataset(IterableDataset[T_co]):
    r""" :class:`SamplerIterableDataset`.

    IterableDataset to generate sample elements.
    args:
        dataset: IterableDataset sampled from
        sampler: Sampler class to genereate sample elements from input dataset.
                    Default is :class:`SequentialSampler` for IterableDataset
    """
    dataset: IterableDataset
    sampler: Sampler

    def __init__(self,
                 dataset: IterableDataset,
                 *,
                 sampler: Type[Sampler] = SequentialSampler,
                 **kwargs
                 ) -> None:
        assert isinstance(dataset, Sized), \
            "Sampler class requires input dataset implemented `__len__`"
        self.dataset = dataset
        # https://github.com/python/mypy/pull/9629 will solve
        self.sampler = sampler(data_source=self.dataset, **kwargs)  # type: ignore

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        # Dataset has been tested as `Sized`
        if isinstance(self.sampler, Sized) and len(self.sampler) >= 0:
            return len(self.sampler)
        raise NotImplementedError
