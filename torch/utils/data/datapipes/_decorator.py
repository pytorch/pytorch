import inspect
from functools import wraps
from typing import Any, Callable, Optional, Type, Union, get_type_hints
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.datapipes._typing import _DataPipeMeta


######################################################
# Functional API
######################################################
class functional_datapipe:
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        """
            Args:
                enable_df_api_tracing - if set, any returned DataPipe would accept
                DataFrames API in tracing mode.
        """
        self.name = name
        self.enable_df_api_tracing = enable_df_api_tracing

    def __call__(self, cls):
        if issubclass(cls, IterDataPipe):
            if isinstance(cls, Type):  # type: ignore[arg-type]
                if not isinstance(cls, _DataPipeMeta):
                    raise TypeError('`functional_datapipe` can only decorate IterDataPipe')
            # with non_deterministic decorator
            else:
                if not isinstance(cls, non_deterministic) and \
                    not (hasattr(cls, '__self__') and
                         isinstance(cls.__self__, non_deterministic)):
                    raise TypeError('`functional_datapipe` can only decorate IterDataPipe')
            IterDataPipe.register_datapipe_as_function(self.name, cls, enable_df_api_tracing=self.enable_df_api_tracing)
        elif issubclass(cls, MapDataPipe):
            MapDataPipe.register_datapipe_as_function(self.name, cls)

        return cls


######################################################
# Determinism
######################################################
_determinism: bool = False


class guaranteed_datapipes_determinism:
    prev: bool

    def __init__(self) -> None:
        global _determinism
        self.prev = _determinism
        _determinism = True

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _determinism
        _determinism = self.prev


class non_deterministic:
    cls: Optional[Type[IterDataPipe]] = None
    # TODO: Lambda for picking
    deterministic_fn: Callable[[], bool]

    def __init__(self, arg: Union[Type[IterDataPipe], Callable[[], bool]]) -> None:
        # 1. Decorator doesn't have any argument
        if isinstance(arg, Type):  # type: ignore[arg-type]
            if not issubclass(arg, IterDataPipe):  # type: ignore[arg-type]
                raise TypeError("Only `IterDataPipe` can be decorated with `non_deterministic`"
                                ", but {} is found".format(arg.__name__))
            self.cls = arg  # type: ignore[assignment]
        # 2. Decorator has an argument of a function
        #    This class should behave differently given different inputs. Use this
        #    function to verify the determinism for each instance.
        #    When the function returns True, the instance is non-deterministic. Otherwise,
        #    the instance is a deterministic DataPipe.
        elif isinstance(arg, Callable):  # type:ignore[arg-type]
            self.deterministic_fn = arg  # type: ignore[assignment, misc]
        else:
            raise TypeError("{} can not be decorated by non_deterministic".format(arg))

    def __call__(self, *args, **kwargs):
        global _determinism
        #  Decorate IterDataPipe
        if self.cls is not None:
            if _determinism:
                raise TypeError("{} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. "
                                "You can turn off determinism for this DataPipe if that is acceptable "
                                "for your application".format(self.cls.__name__))
            return self.cls(*args, **kwargs)  # type: ignore[call-arg]

        # Decorate with a functional argument
        if not (isinstance(args[0], Type) and  # type: ignore[arg-type]
                issubclass(args[0], IterDataPipe)):
            raise TypeError("Only `IterDataPipe` can be decorated, but {} is found"
                            .format(args[0].__name__))
        self.cls = args[0]
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        res = self.deterministic_fn(*args, **kwargs)  # type: ignore[call-arg, misc]
        if not isinstance(res, bool):
            raise TypeError("deterministic_fn of `non_deterministic` decorator is required "
                            "to return a boolean value, but {} is found".format(type(res)))
        global _determinism
        if _determinism and res:
            raise TypeError("{} is non-deterministic with the inputs, but you set "
                            "'guaranteed_datapipes_determinism'. You can turn off determinism "
                            "for this DataPipe if that is acceptable for your application"
                            .format(self.cls.__name__))  # type: ignore[union-attr]
        return self.cls(*args, **kwargs)  # type: ignore[call-arg, misc]


######################################################
# Type validation
######################################################
# Validate each argument of DataPipe with hint as a subtype of the hint.
def argument_validation(f):
    signature = inspect.signature(f)
    hints = get_type_hints(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        for argument_name, value in bound.arguments.items():
            if argument_name in hints and isinstance(hints[argument_name], _DataPipeMeta):
                hint = hints[argument_name]
                if not isinstance(value, IterDataPipe):
                    raise TypeError("Expected argument '{}' as a IterDataPipe, but found {}"
                                    .format(argument_name, type(value)))
                if not value.type.issubtype(hint.type):
                    raise TypeError("Expected type of argument '{}' as a subtype of "
                                    "hint {}, but found {}"
                                    .format(argument_name, hint.type, value.type))

        return f(*args, **kwargs)

    return wrapper


# Default value is True
_runtime_validation_enabled: bool = True


class runtime_validation_disabled:
    prev: bool

    def __init__(self) -> None:
        global _runtime_validation_enabled
        self.prev = _runtime_validation_enabled
        _runtime_validation_enabled = False

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _runtime_validation_enabled
        _runtime_validation_enabled = self.prev


# Runtime checking
# Validate output data is subtype of return hint
def runtime_validation(f):
    # TODO:
    # Can be extended to validate '__getitem__' and nonblocking
    if f.__name__ != '__iter__':
        raise TypeError("Can not decorate function {} with 'runtime_validation'"
                        .format(f.__name__))

    @wraps(f)
    def wrapper(self):
        global _runtime_validation_enabled
        if not _runtime_validation_enabled:
            yield from f(self)
        else:
            it = f(self)
            for d in it:
                if not self.type.issubtype_of_instance(d):
                    raise RuntimeError("Expected an instance as subtype of {}, but found {}({})"
                                       .format(self.type, d, type(d)))
                yield d

    return wrapper
