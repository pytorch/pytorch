import random
import torch

from torch.utils.data import Sampler, SequentialSampler
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from typing import Dict, Iterator, List, Optional, Sized, Tuple, Type, TypeVar

__all__ = [
    "SamplerIterDataPipe",
    "ShufflerIterDataPipe",
]

T_co = TypeVar('T_co', covariant=True)


class SamplerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Generates sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).

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
        if isinstance(self.sampler, Sized):
            return len(self.sampler)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe('shuffle')
class ShufflerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Shuffles the input DataPipe with a buffer (functional name: ``shuffle``). The buffer
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

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> shuffle_dp = dp.shuffle()
        >>> list(shuffle_dp)
        [0, 4, 1, 6, 3, 2, 9, 5, 7, 8]
    """
    datapipe: IterDataPipe[T_co]
    buffer_size: int
    _buffer: List[T_co]
    _enabled: bool
    _seed: Optional[int]
    _rng: random.Random

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 *,
                 buffer_size: int = 10000,
                 unbatch_level: int = 0
                 ) -> None:
        super().__init__()
        # TODO: Performance optimization
        #       buffer can be a fixed size and remove expensive `append()` and `len()` operations
        self._buffer: List[T_co] = []
        assert buffer_size > 0, "buffer_size should be larger than 0"
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.buffer_size = buffer_size
        self._enabled = True
        self._seed = None
        self._rng = random.Random()

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[T_co]:
        if not self._enabled:
            for x in self.datapipe:
                yield x
        else:
            for x in self.datapipe:
                if len(self._buffer) == self.buffer_size:
                    idx = self._rng.randint(0, len(self._buffer) - 1)
                    val, self._buffer[idx] = self._buffer[idx], x
                    yield val
                else:
                    self._buffer.append(x)
            while self._buffer:
                idx = self._rng.randint(0, len(self._buffer) - 1)
                yield self._buffer.pop(idx)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))

    def reset(self) -> None:
        self._buffer = []
        if self._enabled:
            if self._seed is None:
                self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self._rng.seed(self._seed)
            self._seed = None

    def __getstate__(self):
        state = (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            self._rng.getstate(),
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    def __del__(self):
        self._buffer.clear()
