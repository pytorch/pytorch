# mypy: allow-untyped-defs
from typing import Callable, Iterator, Tuple, TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import (
    _check_unpickable_fn,
    StreamWrapper,
    validate_input_col,
)


__all__ = ["FilterIterDataPipe"]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("filter")
class FilterIterDataPipe(IterDataPipe[_T_co]):
    r"""
    Filters out elements from the source datapipe according to input ``filter_fn`` (functional name: ``filter``).

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        input_col: Index or indices of data which ``filter_fn`` is applied, such as:

            - ``None`` as default to apply ``filter_fn`` to the data directly.
            - Integer(s) is used for list/tuple.
            - Key(s) is used for dict.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def is_even(n):
        ...     return n % 2 == 0
        >>> dp = IterableWrapper(range(5))
        >>> filter_dp = dp.filter(filter_fn=is_even)
        >>> list(filter_dp)
        [0, 2, 4]
    """

    datapipe: IterDataPipe[_T_co]
    filter_fn: Callable

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        filter_fn: Callable,
        input_col=None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe

        _check_unpickable_fn(filter_fn)
        self.filter_fn = filter_fn

        self.input_col = input_col
        validate_input_col(filter_fn, input_col)

    def _apply_filter_fn(self, data) -> bool:
        if self.input_col is None:
            return self.filter_fn(data)
        elif isinstance(self.input_col, (list, tuple)):
            args = tuple(data[col] for col in self.input_col)
            return self.filter_fn(*args)
        else:
            return self.filter_fn(data[self.input_col])

    def __iter__(self) -> Iterator[_T_co]:
        for data in self.datapipe:
            condition, filtered = self._returnIfTrue(data)
            if condition:
                yield filtered
            else:
                StreamWrapper.close_streams(data)

    def _returnIfTrue(self, data: _T) -> Tuple[bool, _T]:
        condition = self._apply_filter_fn(data)

        if df_wrapper.is_column(condition):
            # We are operating on DataFrames filter here
            result = []
            for idx, mask in enumerate(df_wrapper.iterate(condition)):
                if mask:
                    result.append(df_wrapper.get_item(data, idx))
            if len(result):
                return True, df_wrapper.concat(result)
            else:
                return False, None  # type: ignore[return-value]

        if not isinstance(condition, bool):
            raise ValueError(
                "Boolean output is required for `filter_fn` of FilterIterDataPipe, got",
                type(condition),
            )

        return condition, data
