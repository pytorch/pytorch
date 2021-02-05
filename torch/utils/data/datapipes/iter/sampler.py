from torch.utils.data import IterDataPipe, Sampler, SequentialSampler
from typing import Callable, TypeVar, Type, Iterator, Sized

from .callable import CallableIterDataPipe

T_co = TypeVar('T_co', covariant=True)


class FilterIterDataPipe(CallableIterDataPipe[T_co]):
    r""" :class:`FilterIterDataPipe`.

    Iterable DataPipe to filter elements from datapipe according to filter_fn.
    args:
        datapipe: Iterable DataPipe being filterd
        filter_fn: Customized function mapping an element to a boolean.
    """
    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 *args,
                 filter_fn: Callable[..., bool],
                 **kwargs,
                 ) -> None:
        super().__init__(datapipe, *args, fn=filter_fn, **kwargs)

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            if (self.fn(data, *self.args, **self.kwargs)):
                yield data

    def __len__(self):
        raise(NotImplementedError)


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
                 *,
                 sampler: Type[Sampler] = SequentialSampler,
                 **kwargs
                 ) -> None:
        assert isinstance(datapipe, Sized), \
            "Sampler class requires input datapipe implemented `__len__`"
        super().__init__()
        self.datapipe = datapipe
        # https://github.com/python/mypy/pull/9629 will solve
        self.sampler = sampler(data_source=self.datapipe, **kwargs)  # type: ignore

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        # Dataset has been tested as `Sized`
        if isinstance(self.sampler, Sized) and len(self.sampler) >= 0:
            return len(self.sampler)
        raise NotImplementedError
