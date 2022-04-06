from typing import Callable, Iterator, TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.utils.common import check_lambda_fn

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('filter')
class FilterIterDataPipe(IterDataPipe[T_co]):
    r"""
    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        drop_empty_batches: By default, drops a batch if it is empty after filtering instead of keeping an empty list

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    """
    datapipe: IterDataPipe
    filter_fn: Callable
    drop_empty_batches: bool

    def __init__(self,
                 datapipe: IterDataPipe,
                 filter_fn: Callable,
                 drop_empty_batches: bool = True,
                 ) -> None:
        super().__init__()
        self.datapipe = datapipe
        check_lambda_fn(filter_fn)

        self.filter_fn = filter_fn  # type: ignore[assignment]
        self.drop_empty_batches = drop_empty_batches

    def __iter__(self) -> Iterator[T_co]:
        res: bool
        for data in self.datapipe:
            filtered = self._returnIfTrue(data)
            if self._isNonEmpty(filtered):
                yield filtered

    def _returnIfTrue(self, data):
        condition = self.filter_fn(data)

        if df_wrapper.is_column(condition):
            # We are operating on DataFrames filter here
            result = []
            for idx, mask in enumerate(df_wrapper.iterate(condition)):
                if mask:
                    result.append(df_wrapper.get_item(data, idx))
            if len(result):
                return df_wrapper.concat(result)
            else:
                return None

        if not isinstance(condition, bool):
            raise ValueError("Boolean output is required for `filter_fn` of FilterIterDataPipe, got", type(condition))
        if condition:
            return data

    def _isNonEmpty(self, data):
        if df_wrapper.is_dataframe(data):
            return True
        r = data is not None and \
            not (isinstance(data, list) and len(data) == 0 and self.drop_empty_batches)
        return r
