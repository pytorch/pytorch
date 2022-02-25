import warnings
from typing import Callable, TypeVar

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
    r"""
    Apply the input function over each item from the source DataPipe (functional name: ``map``).
    The function can be any regular Python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.

    Args:
        datapipe: Source MapDataPipe
        fn: Function being applied to each item
    """
    datapipe: MapDataPipe
    fn: Callable

    def __init__(
        self,
        datapipe: MapDataPipe,
        fn: Callable = default_fn,
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

    def __len__(self) -> int:
        return len(self.datapipe)

    def __getitem__(self, index) -> T_co:
        return self.fn(self.datapipe[index])

    def __getstate__(self):
        if DILL_AVAILABLE:
            dill_function = dill.dumps(self.fn)
        else:
            dill_function = self.fn
        state = (self.datapipe, dill_function)
        return state

    def __setstate__(self, state):
        (self.datapipe, dill_function) = state
        if DILL_AVAILABLE:
            self.fn = dill.loads(dill_function)  # type: ignore[assignment]
        else:
            self.fn = dill_function  # type: ignore[assignment]
