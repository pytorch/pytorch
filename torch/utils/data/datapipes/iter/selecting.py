from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Callable, TypeVar, Iterator, Optional, Tuple, Dict

from .callable import MapIterDataPipe

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('filter')
class FilterIterDataPipe(MapIterDataPipe):
    r""" :class:`FilterIterDataPipe`.

    Iterable DataPipe to filter elements from datapipe according to filter_fn.
    args:
        datapipe: Iterable DataPipe being filterd
        filter_fn: Customized function mapping an element to a boolean.
        fn_args: Positional arguments for `filter_fn`
        fn_kwargs: Keyword arguments for `filter_fn`
    """
    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 filter_fn: Callable[..., bool],
                 fn_args: Optional[Tuple] = None,
                 fn_kwargs: Optional[Dict] = None,
                 batch_level: bool = False,
                 drop_empty_batches: bool = True,
                 ) -> None:
        self.drop_empty_batches = drop_empty_batches
        super().__init__(datapipe, fn=filter_fn, fn_args=fn_args, fn_kwargs=fn_kwargs, batch_level = batch_level)

    def __iter__(self) -> Iterator[T_co]:
        res: bool
        for data in self.datapipe:
            if self.batch_level or not isinstance(data, list):
                res = self.fn(data, *self.args, **self.kwargs)
                if not isinstance(res, bool):
                    raise ValueError("Boolean output is required for "
                                    "`filter_fn` of FilterIterDataPipe")
                if res:
                    yield data
            else:
                res = []
                for el in data:
                    if self.fn(el, *self.args, **self.kwargs):
                        res.append(el)
                if len(res) or not self.drop_empty_batches:
                    yield res
            

    def __len__(self):
        raise(NotImplementedError)
