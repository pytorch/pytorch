from torch.utils.data import IterDataPipe
from typing import Callable, TypeVar, Iterator

from .callable import MapIterDataPipe

T_co = TypeVar('T_co', covariant=True)


class FilterIterDataPipe(MapIterDataPipe[T_co]):
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
        res: bool
        for data in self.datapipe:
            res = self.fn(data, *self.args, **self.kwargs)
            if not isinstance(res, bool):
                raise ValueError("Boolean output is required for "
                                 "`filter_fn` of FilterIterDataPipe")
            if res:
                yield data

    def __len__(self):
        raise(NotImplementedError)
