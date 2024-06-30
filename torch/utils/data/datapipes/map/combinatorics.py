# mypy: allow-untyped-defs
import random
from typing import Iterator, List, Optional, TypeVar

import torch
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


__all__ = ["ShufflerIterDataPipe"]


T_co = TypeVar("T_co", covariant=True)


# @functional_datapipe('shuffle')
class ShufflerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> shuffle_dp = dp.shuffle().set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
        >>> list(shuffle_dp)
        [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]
        >>> # Reset seed for Shuffler
        >>> shuffle_dp = shuffle_dp.set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    Note:
        Even thought this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an
        ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to
        the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order
        of data during data-processing.
    """

    datapipe: MapDataPipe[T_co]
    _enabled: bool
    _seed: Optional[int]
    _rng: random.Random

    def __init__(
        self,
        datapipe: MapDataPipe[T_co],
        *,
        indices: Optional[List] = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.indices = list(range(len(datapipe))) if indices is None else indices
        self._enabled = True
        self._seed = None
        self._rng = random.Random()
        self._shuffled_indices: List = self.indices

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[T_co]:
        if not self._enabled:
            for idx in self.indices:
                yield self.datapipe[idx]
        else:
            while self._shuffled_indices:
                idx = self._shuffled_indices.pop()
                yield self.datapipe[idx]

    def reset(self) -> None:
        if self._enabled and self._seed is None:
            self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self._rng.seed(self._seed)
        self._seed = None
        self._shuffled_indices = self._rng.sample(self.indices, len(self.indices))

    def __len__(self) -> int:
        return len(self.datapipe)

    def __getstate__(self):
        state = (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            self._rng.getstate(),
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            rng_state,
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)


MapDataPipe.register_datapipe_as_function("shuffle", ShufflerIterDataPipe)
