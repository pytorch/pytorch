import random

from torch.utils.data import MapDataPipe, functional_datapipe
from typing import Dict, List, Optional, TypeVar


T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('shuffle')
class ShufflerMapDataPipe(MapDataPipe[T_co]):
    r""" :class:`ShufflerMapDataPipe`

    Map DataPipe to shuffle the input DataPipe via its indices.

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
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
        self.possible_indices = set(range(len(datapipe))) if indices is None else set(indices)
        self.index_mapping: Dict = {}

    def __getitem__(self, index):
        if index in self.index_mapping:
            return self.datapipe[self.index_mapping[index]]
        else:
            try:
                new_idx = random.sample(self.possible_indices, 1)[0]
            except ValueError:
                raise IndexError(f"Index {index} is out of bound.")
            self.possible_indices.remove(new_idx)
            self.index_mapping[index] = new_idx
            return self.datapipe[new_idx]

    def __len__(self) -> int:
        return len(self.datapipe)
