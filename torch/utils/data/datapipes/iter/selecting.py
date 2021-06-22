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
        nesting_level: Determines which level the fn gets applied to, by default it applies to the top level (= 0)
    """
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

    def _merge(self, data, mask):
        result = []
        for i, b in zip(data, mask):
            if isinstance(b, list):
                t = self._merge(i, b)
                if len(t) > 0 or not self.drop_empty_batches:
                    result.append(t)
            else:
                if not isinstance(b, bool):
                    raise ValueError("Boolean output is required for "
                                     "`filter_fn` of FilterIterDataPipe")
                if b:
                    result.append(i)
        return result

    def __iter__(self) -> Iterator[T_co]:
        res: bool
        for data in self.datapipe:
            if self.nesting_level == 0:
                res = self.fn(data, *self.args, **self.kwargs)
                if not isinstance(res, bool):
                    raise ValueError("Boolean output is required for "
                                     "`filter_fn` of FilterIterDataPipe")
                if res:
                    yield data
            else:
                mask = self._apply(data, self.nesting_level)
                merged = self._merge(data, mask)
                if len(merged) > 0 or not self.drop_empty_batches:
                    yield merged

    def __len__(self):
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
