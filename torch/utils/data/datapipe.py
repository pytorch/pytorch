import functools
from typing import Dict, Callable, Optional

from torch.utils.data._typing import _DataPipeMeta
from torch.utils.data._utils.serialization import serialize_fn, SerializationType, deserialize_fn
from torch.utils.data.dataset import IterableDataset, T_co, UNTRACABLE_DATAFRAME_PIPES, Dataset


class IterDataPipe(IterableDataset[T_co], metaclass=_DataPipeMeta):
    r"""
    Iterable-style DataPipe.

    All DataPipes that represent an iterable of data samples should subclass this.
    This style of DataPipes is particularly useful when data come from a stream, or
    when the number of samples is too large to fit them all in memory.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this DataPipe.

    `IterDataPipe` is lazily initialized and its elements are computed only when ``next()`` is called
    on its iterator.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `IterDataPipe` (recommended, available to most but not all DataPipes).
    You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
    operations in succession.

    Note:
        When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
        item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
        iterator. When :attr:`num_workers > 0`, each worker process will have a
        different copy of the DataPipe object, so it is often desired to configure
        each copy independently to avoid having duplicate data returned from the
        workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
        process, returns information about the worker. It can be used in either the
        dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
        :attr:`worker_init_fn` option to modify each copy's behavior.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> dp = IterableWrapper(range(10))
        >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> map_dp_2 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
        >>> list(filter_dp)
        [2, 4, 6, 8, 10]
    """
    functions: Dict[str, Callable] = {}
    reduce_ex_hook : Optional[Callable] = None
    getstate_hook: Optional[Callable] = None

    def __getattr__(self, attribute_name):
        if attribute_name in IterDataPipe.functions:
            function = functools.partial(IterDataPipe.functions[attribute_name], self)
            return function
        else:
            raise AttributeError("'{0}' object has no attribute '{1}".format(self.__class__.__name__, attribute_name))

    @classmethod
    def register_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register, enable_df_api_tracing=False):
        if function_name in cls.functions:
            raise Exception("Unable to add DataPipe function name {} as it is already taken".format(function_name))

        def class_function(cls, enable_df_api_tracing, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            if isinstance(result_pipe, IterDataPipe):
                if enable_df_api_tracing or isinstance(source_dp, DFIterDataPipe):
                    if function_name not in UNTRACABLE_DATAFRAME_PIPES:
                        result_pipe = result_pipe.trace_as_dataframe()

            return result_pipe

        function = functools.partial(class_function, cls_to_register, enable_df_api_tracing)
        cls.functions[function_name] = function

    def __getstate__(self):
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(self)
        state_dict = {}
        for k, v in self.__dict__.items():
            if callable(v):
                state_dict[k] = serialize_fn(v)
            else:
                state_dict[k] = v
        return state_dict

    def __setstate__(self, state_dict):
        for k, v in state_dict.items():
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], SerializationType):
                self.__dict__[k] = deserialize_fn(v)
            else:
                self.__dict__[k] = v

    def __reduce_ex__(self, *args, **kwargs):
        if IterDataPipe.reduce_ex_hook is not None:
            try:
                return IterDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        if IterDataPipe.getstate_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing getstate_hook")
        IterDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn):
        if IterDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing reduce_ex_hook")
        IterDataPipe.reduce_ex_hook = hook_fn


class DFIterDataPipe(IterDataPipe):
    def _is_dfpipe(self):
        return True


class MapDataPipe(Dataset[T_co], metaclass=_DataPipeMeta):
    r"""
    Map-style DataPipe.

    All datasets that represent a map from keys to data samples should subclass this.
    Subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given, unique key. Subclasses can also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    These DataPipes can be invoked in two ways, using the class constructor or applying their
    functional form onto an existing `MapDataPipe` (recommend, available to most but not all DataPipes).

    Note:
        :class:`~torch.utils.data.DataLoader` by default constructs an index
        sampler that yields integral indices. To make it work with a map-style
        DataPipe with non-integral indices/keys, a custom sampler must be provided.

    Example:
        >>> from torchdata.datapipes.map import SequenceWrapper, Mapper
        >>> dp = SequenceWrapper(range(10))
        >>> map_dp_1 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> map_dp_2 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> batch_dp = map_dp_1.batch(batch_size=2)
        >>> list(batch_dp)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    """
    functions: Dict[str, Callable] = {}

    def __getattr__(self, attribute_name):
        if attribute_name in MapDataPipe.functions:
            function = functools.partial(MapDataPipe.functions[attribute_name], self)
            return function
        else:
            raise AttributeError("'{0}' object has no attribute '{1}".format(self.__class__.__name__, attribute_name))

    @classmethod
    def register_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in cls.functions:
            raise Exception("Unable to add DataPipe function name {} as it is already taken".format(function_name))

        def class_function(cls, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            return result_pipe

        function = functools.partial(class_function, cls_to_register)
        cls.functions[function_name] = function

    def __getstate__(self):
        state_dict = {}
        for k, v in self.__dict__.items():
            if callable(v):
                state_dict[k] = serialize_fn(v)
            else:
                state_dict[k] = v
        return state_dict

    def __setstate__(self, state_dict):
        for k, v in state_dict.items():
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], SerializationType):
                self.__dict__[k] = deserialize_fn(v)
            else:
                self.__dict__[k] = v
