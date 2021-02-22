from torch.utils.data import IterDataPipe, Sampler, SequentialSampler
from typing import TypeVar, Type, Iterator, Sized, Optional, Tuple, Dict

T_co = TypeVar('T_co', covariant=True)


class SamplerIterDataPipe(IterDataPipe[T_co]):
    r""" :class:`SamplerIterDataPipe`.

    Iterable DataPipe to generate sample elements.
    args:
        datapipe: IterDataPipe sampled from
        sampler: Sampler class to genereate sample elements from input DataPipe.
                    Default is :class:`SequentialSampler` for IterDataPipe
    """
    datapipe: IterDataPipe
    sampler: Sampler

    def __init__(self,
                 datapipe: IterDataPipe,
                 sampler: Type[Sampler] = SequentialSampler,
                 sampler_args: Optional[Tuple] = None,
                 sampler_kwargs: Optional[Dict] = None
                 ) -> None:
        assert isinstance(datapipe, Sized), \
            "Sampler class requires input datapipe implemented `__len__`"
        super().__init__()
        self.datapipe = datapipe
        self.sampler_args = () if sampler_args is None else sampler_args
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs
        # https://github.com/python/mypy/pull/9629 will solve
        self.sampler = sampler(data_source=self.datapipe, *self.sampler_args, **self.sampler_kwargs)  # type: ignore

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        # Dataset has been tested as `Sized`
        if isinstance(self.sampler, Sized) and len(self.sampler) >= 0:
            return len(self.sampler)
        raise NotImplementedError
