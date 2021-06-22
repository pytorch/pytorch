from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Callable, TypeVar, Iterator, Optional, Tuple, Dict

from .callable import MapIterDataPipe

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('filter')
class FilterIterDataPipe(MapIterDataPipe):
    r""" :class:`FilterIterDataPipe`.

    Iterable DataPipe to filter elements from datapipe according to filter_fn.
    args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        fn_args: Positional arguments for `filter_fn`
        fn_kwargs: Keyword arguments for `filter_fn`
        drop_empty_batches: By default, drops batch if it is empty after filtering instead of keeping an empty list
        nesting_level: Determines which level the fn gets applied to, by default it applies to the top level (= 0).
        This also accepts -1 as input to apply filtering to the lowest nesting level. It currently doesn't support
        argument < -1.
    """
    drop_empty_batches: bool

    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 filter_fn: Callable[..., bool],
                 fn_args: Optional[Tuple] = None,
                 fn_kwargs: Optional[Dict] = None,
                 drop_empty_batches: bool = True,
                 nesting_level: int = 0,
                 ) -> None:
        self.drop_empty_batches = drop_empty_batches
        super().__init__(datapipe, fn=filter_fn, fn_args=fn_args, fn_kwargs=fn_kwargs, nesting_level=nesting_level)

    def __iter__(self) -> Iterator[T_co]:
        res: bool
        for data in self.datapipe:
            filtered = self._applyFilter(data, self.nesting_level)
            if self._isNonEmpty(filtered):
                yield filtered

    def _applyFilter(self, data, nesting_level):
        if nesting_level == 0:
            return self._returnIfTrue(data)
        elif nesting_level > 0:
            if not isinstance(data, list):
                raise IndexError(f"nesting_level {self.nesting_level} out of range (exceeds data pipe depth)")
            result = filter(self._isNonEmpty, [self._applyFilter(i, nesting_level - 1) for i in data])
            return list(result)
        else:  # Handling nesting_level == -1
            if isinstance(data, list):
                result = filter(self._isNonEmpty, [self._applyFilter(i, nesting_level) for i in data])
                return list(result)
            else:
                return self._returnIfTrue(data)

    def _returnIfTrue(self, data):
        condition = self.fn(data, *self.args, **self.kwargs)
        if not isinstance(condition, bool):
            raise ValueError("Boolean output is required for `filter_fn` of FilterIterDataPipe")
        if condition:
            return data

    def _isNonEmpty(self, data):
        return data is not None and not (data == [] and self.drop_empty_batches)

    def __len__(self):
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
