import random

from torch.utils.data import IterDataPipe, Sampler, SequentialSampler, functional_datapipe
from typing import Dict, Iterator, List, Optional, Sized, Tuple, Type, TypeVar

T_co = TypeVar('T_co', covariant=True)


class SamplerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Generate sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).

    Args:
        datapipe: IterDataPipe to sample from
        sampler: Sampler class to generate sample elements from input DataPipe.
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
        self.sampler = sampler(data_source=self.datapipe, *self.sampler_args, **self.sampler_kwargs)  # type: ignore[misc]

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)

    def __len__(self) -> int:
        # Dataset has been tested as `Sized`
        if isinstance(self.sampler, Sized) and len(self.sampler) >= 0:
            return len(self.sampler)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe('shuffle')
class ShufflerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Shuffle the input DataPipe with a buffer (functional name: ``shuffle``). The buffer
    with ``buffer_size`` is filled with elements from the datapipe first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.

    ``buffer_size`` is required to be larger than ``0``. For ``buffer_size == 1``, the
    datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
    ``buffer_size`` is required to be greater than or equal to the size of datapipe.

    When it is used with :class:`torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: The IterDataPipe being shuffled
        buffer_size: The buffer size for shuffling (default to ``10000``)
        unbatch_level: Specifies if it is necessary to unbatch source data before
            applying the shuffle
    """
    datapipe: IterDataPipe[T_co]
    buffer_size: int
    _shuffle_enabled: bool

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 *,
                 default: bool = True,
                 buffer_size: int = 10000,
                 unbatch_level: int = 0
                 ) -> None:
        super().__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.buffer_size = buffer_size
        self._shuffle_enabled = default

    @staticmethod
    def buffer_replace(buffer, x):
        idx = random.randint(0, len(buffer) - 1)
        val = buffer[idx]
        buffer[idx] = x
        return val

    def set_shuffle_settings(self, shuffle=True):
        self._shuffle_enabled = shuffle

    def __iter__(self) -> Iterator[T_co]:
        if not self._shuffle_enabled:
            for x in self.datapipe:
                yield x
        else:
            buffer: List[T_co] = []
            for x in self.datapipe:
                if len(buffer) == self.buffer_size:
                    yield ShufflerIterDataPipe.buffer_replace(buffer, x)
                else:
                    buffer.append(x)
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
