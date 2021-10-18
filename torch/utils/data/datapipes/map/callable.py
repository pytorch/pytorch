import warnings
from typing import Callable, Dict, Optional, Tuple, TypeVar

from torch.utils.data import MapDataPipe, functional_datapipe

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

T_co = TypeVar('T_co', covariant=True)


# Default function to return each item directly
# In order to keep datapipe picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    return data


@functional_datapipe('map')
class MapperMapDataPipe(MapDataPipe[T_co]):
    r""":class:`MapperMapDataPipe`.

    Map DataPipe to run a function over each item from the source DataPipe.
    The function can be any regular python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.
    args:
        datapipe: Source Map DataPipe
        fn: Function called over each item
        fn_args: Positional arguments for `fn`
        fn_kwargs: Keyword arguments for `fn`
    """
    datapipe: MapDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: MapDataPipe,
        fn: Callable = default_fn,
        fn_args: Optional[Tuple] = None,
        fn_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        # Partial object has no attribute '__name__', but can be pickled
        if hasattr(fn, '__name__') and fn.__name__ == '<lambda>' and not DILL_AVAILABLE:
            warnings.warn(
                "Lambda function is not supported for pickle, please use "
                "regular python function or functools.partial instead."
            )
        self.fn = fn  # type: ignore[assignment]
        self.args = () if fn_args is None else fn_args
        self.kwargs = {} if fn_kwargs is None else fn_kwargs

    def __len__(self) -> int:
        return len(self.datapipe)

    def __getitem__(self, index) -> T_co:
        return self.fn(self.datapipe[index], *self.args, **self.kwargs)

    def __getstate__(self):
        if DILL_AVAILABLE:
            dill_function = dill.dumps(self.fn)
        else:
            dill_function = self.fn
        state = (self.datapipe, dill_function, self.args, self.kwargs)
        return state

    def __setstate__(self, state):
        (self.datapipe, dill_function, self.args, self.kwargs) = state
        if DILL_AVAILABLE:
            self.fn = dill.loads(dill_function)  # type: ignore[assignment]
        else:
            self.fn = dill_function  # type: ignore[assignment]
