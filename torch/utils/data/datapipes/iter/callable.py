import warnings
import torch.nn as nn
from torch.utils.data import IterDataPipe, _utils, functional_datapipe
from typing import Callable, Dict, Iterator, Optional, Sized, Tuple, TypeVar

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
class MapIterDataPipe(IterDataPipe[T_co]):
    r""" :class:`MapIterDataPipe`.

    Iterable DataPipe to run a function over each item from the source DataPipe.
    The function can be any regular python function or partial object. Lambda
    function is not recommended as it is not supported by pickle.
    args:
        datapipe: Source Iterable DataPipe
        fn: Function called over each item
        fn_args: Positional arguments for `fn`
        fn_kwargs: Keyword arguments for `fn`
    """
    datapipe: IterDataPipe
    fn: Callable

    def __init__(self,
                 datapipe: IterDataPipe,
                 fn: Callable = default_fn,
                 fn_args: Optional[Tuple] = None,
                 fn_kwargs: Optional[Dict] = None,
                 ) -> None:
        super().__init__()
        self.datapipe = datapipe
        # Partial object has no attribute '__name__', but can be pickled
        if hasattr(fn, '__name__') and fn.__name__ == '<lambda>' and not DILL_AVAILABLE:
            warnings.warn("Lambda function is not supported for pickle, please use "
                          "regular python function or functools.partial instead.")
        self.fn = fn  # type: ignore
        self.args = () if fn_args is None else fn_args
        self.kwargs = {} if fn_kwargs is None else fn_kwargs

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            yield self.fn(data, *self.args, **self.kwargs)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError

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
            self.fn = dill.loads(dill_function)  # type: ignore
        else:
            self.fn = dill_function  # type: ignore


@functional_datapipe('collate')
class CollateIterDataPipe(MapIterDataPipe):
    r""" :class:`CollateIterDataPipe`.

    Iterable DataPipe to collate samples from datapipe to Tensor(s) by `util_.collate.default_collate`,
    or customized Data Structure by collate_fn.
    args:
        datapipe: Iterable DataPipe being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
                    Default function collates to Tensor(s) based on data type.
        fn_args: Positional arguments for `collate_fn`
        fn_kwargs: Keyword arguments for `collate_fn`

    Example: Convert integer data to float Tensor
        >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
        ...     def __init__(self, start, end):
        ...         super(MyIterDataPipe).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        ...     def __len__(self):
        ...         return self.end - self.start
        ...
        >>> ds = MyIterDataPipe(start=3, end=7)
        >>> print(list(ds))
        [3, 4, 5, 6]

        >>> def collate_fn(batch):
        ...     return torch.tensor(batch, dtype=torch.float)
        ...
        >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
        >>> print(list(collated_ds))
        [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
    """
    def __init__(self,
                 datapipe: IterDataPipe,
                 collate_fn: Callable = _utils.collate.default_collate,
                 fn_args: Optional[Tuple] = None,
                 fn_kwargs: Optional[Dict] = None,
                 ) -> None:
        super().__init__(datapipe, fn=collate_fn, fn_args=fn_args, fn_kwargs=fn_kwargs)


@functional_datapipe('transforms')
class TransformsIterDataPipe(MapIterDataPipe):
    r""" :class:`TransformsIterDataPipe`.

    Iterable DataPipe to use transform(s) from torchvision or torchaudio to transform
    data from datapipe.
    args:
        datapipe: Iterable DataPipe being transformed
        transforms: A transform or a sequence of transforms from torchvision or torchaudio.
    """
    def __init__(self,
                 datapipe: IterDataPipe,
                 transforms: Callable,
                 ) -> None:
        # Type checking for transforms
        transforms_types: Tuple = (nn.Module, )
        try:
            # Specific types of transforms other than `nn.Module` from torchvision
            import torchvision.transforms as tsfm
            transforms_types += (tsfm.Compose, tsfm.RandomChoice, tsfm.RandomOrder,
                                 tsfm.ToPILImage, tsfm.ToTensor, tsfm.Lambda)
        except ImportError:
            pass

        if not isinstance(transforms, transforms_types):
            raise TypeError("`transforms` are required to be a callable from "
                            "torchvision.transforms or torchaudio.transforms")

        super().__init__(datapipe, fn=transforms)
