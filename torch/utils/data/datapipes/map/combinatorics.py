import random

from torch.utils.data import MapDataPipe, functional_datapipe
from typing import Iterator, List, Optional, TypeVar


T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('shuffle')
class ShufflerMapDataPipe(MapDataPipe[T_co]):
    r"""
    Shuffle the input DataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing
    """
    datapipe: MapDataPipe[T_co]

    def __init__(self,
                 datapipe: MapDataPipe[T_co],
                 *,
                 indices: Optional[List] = None,
                 ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.indices = list(range(len(datapipe))) if indices is None else indices
        self.index_map = {index_name: num_index for num_index, index_name in enumerate(self.indices)}
        # We do not lazily shuffle because this way is significantly faster in terms of total time
        random.shuffle(self.indices)

    def __getitem__(self, index) -> T_co:
        old_numeric_index = self.index_map[index]
        new_index = self.indices[old_numeric_index]
        return self.datapipe[new_index]

    # Without __iter__ implemented, by default it tries to use 0-index,
    # which doesn't work when there is a custom index.
    def __iter__(self) -> Iterator[T_co]:
        for i in self.indices:
            yield self.datapipe[i]

    def __len__(self) -> int:
        return len(self.datapipe)
