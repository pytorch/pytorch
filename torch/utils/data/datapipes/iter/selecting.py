import warnings
from typing import Callable, Dict, Iterator, Optional, Tuple, TypeVar

from torch.utils.data import DataChunk, IterDataPipe, functional_datapipe

try:
    import pandas  # type: ignore[import]
    # pandas used only for prototyping, will be shortly replaced with TorchArrow
    WITH_PANDAS = True
except ImportError:
    WITH_PANDAS = False

T_co = TypeVar('T_co', covariant=True)

try:
    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False


@functional_datapipe('filter')
class FilterIterDataPipe(IterDataPipe[T_co]):
    r""" :class:`FilterIterDataPipe`.

    Iterable DataPipe to filter elements from datapipe according to filter_fn.

    Args:
        datapipe: Iterable DataPipe being filtered
        filter_fn: Customized function mapping an element to a boolean.
        fn_args: Positional arguments for `filter_fn`
        fn_kwargs: Keyword arguments for `filter_fn`
        drop_empty_batches: By default, drops batch if it is empty after filtering instead of keeping an empty list
        nesting_level: Determines which level the fn gets applied to, by default it applies to the top level (= 0).
            This also accepts -1 as input to apply filtering to the lowest nesting level.
            It currently doesn't support argument < -1.
    """
    datapipe: IterDataPipe
    filter_fn: Callable
    drop_empty_batches: bool

    def __init__(self,
                 datapipe: IterDataPipe,
                 filter_fn: Callable,
                 fn_args: Optional[Tuple] = None,
                 fn_kwargs: Optional[Dict] = None,
                 drop_empty_batches: bool = True,
                 nesting_level: int = 0,
                 ) -> None:
        super().__init__()
        self.datapipe = datapipe
        # Partial object has no attribute '__name__', but can be pickled
        if hasattr(filter_fn, '__name__') and filter_fn.__name__ == '<lambda>' and not DILL_AVAILABLE:
            warnings.warn("Lambda function is not supported for pickle, please use "
                          "regular python function or functools.partial instead.")
        self.filter_fn = filter_fn  # type: ignore[assignment]
        self.args = () if fn_args is None else fn_args
        self.kwargs = {} if fn_kwargs is None else fn_kwargs
        if nesting_level < -1:
            raise ValueError("nesting_level must be -1 or >= 0")
        self.nesting_level = nesting_level
        self.drop_empty_batches = drop_empty_batches

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
            if isinstance(data, DataChunk):
                result = filter(self._isNonEmpty, [self._applyFilter(i, nesting_level - 1)
                                                   for i in data.raw_iterator()])
                return type(data)(list(result))
            elif isinstance(data, list):
                result = filter(self._isNonEmpty, [self._applyFilter(i, nesting_level - 1) for i in data])
                return list(result)
            else:
                raise IndexError(f"nesting_level {self.nesting_level} out of range (exceeds data pipe depth)")
        else:  # Handling nesting_level == -1
            if isinstance(data, DataChunk):
                result = filter(self._isNonEmpty, [self._applyFilter(i, nesting_level) for i in data.raw_iterator()])
                return type(data)(list(result))
            elif isinstance(data, list):
                result = filter(self._isNonEmpty, [self._applyFilter(i, nesting_level) for i in data])
                return list(result)
            else:
                return self._returnIfTrue(data)

    def _returnIfTrue(self, data):
        condition = self.filter_fn(data, *self.args, **self.kwargs)
        if WITH_PANDAS:
            if isinstance(condition, pandas.core.series.Series):
                # We are operatring on DataFrames filter here
                result = []
                for idx, mask in enumerate(condition):
                    if mask:
                        result.append(data[idx:idx + 1])
                if len(result):
                    return pandas.concat(result)
                else:
                    return None

        if not isinstance(condition, bool):
            raise ValueError("Boolean output is required for `filter_fn` of FilterIterDataPipe, got", type(condition))
        if condition:
            return data

    def _isNonEmpty(self, data):
        if WITH_PANDAS:
            if isinstance(data, pandas.core.frame.DataFrame):
                return True
        r = data is not None and \
            not (isinstance(data, list) and len(data) == 0 and self.drop_empty_batches)
        return r

    def __getstate__(self):
        if DILL_AVAILABLE:
            dill_function = dill.dumps(self.filter_fn)
        else:
            dill_function = self.filter_fn
        state = (self.datapipe, dill_function, self.args, self.kwargs, self.drop_empty_batches, self.nesting_level)
        return state

    def __setstate__(self, state):
        (self.datapipe, dill_function, self.args, self.kwargs, self.drop_empty_batches, self.nesting_level) = state
        if DILL_AVAILABLE:
            self.filter_fn = dill.loads(dill_function)  # type: ignore[assignment]
        else:
            self.filter_fn = dill_function  # type: ignore[assignment]
