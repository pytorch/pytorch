# mypy: allow-untyped-defs
import inspect
from functools import wraps
from typing import Any, Callable, get_type_hints, Optional, Type, Union

from torch.utils.data.datapipes._typing import _DataPipeMeta
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


######################################################
# Functional API
######################################################
class functional_datapipe:
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        """
        Define a functional datapipe.

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
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            # with non_deterministic decorator
            else:
                if not isinstance(cls, non_deterministic) and not (
                    hasattr(cls, "__self__")
                    and isinstance(cls.__self__, non_deterministic)
                ):
                    raise TypeError(
                        "`functional_datapipe` can only decorate IterDataPipe"
                    )
            IterDataPipe.register_datapipe_as_function(
                self.name, cls, enable_df_api_tracing=self.enable_df_api_tracing
            )
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
                raise TypeError(
                    "Only `IterDataPipe` can be decorated with `non_deterministic`"
                    f", but {arg.__name__} is found"
                )
            self.cls = arg  # type: ignore[assignment]
        # 2. Decorator has an argument of a function
        #    This class should behave differently given different inputs. Use this
        #    function to verify the determinism for each instance.
        #    When the function returns True, the instance is non-deterministic. Otherwise,
        #    the instance is a deterministic DataPipe.
        elif isinstance(arg, Callable):  # type:ignore[arg-type]
            self.deterministic_fn = arg  # type: ignore[assignment]
        else:
            raise TypeError(f"{arg} can not be decorated by non_deterministic")

    def __call__(self, *args, **kwargs):
        global _determinism
        #  Decorate IterDataPipe
        if self.cls is not None:
            if _determinism:
                raise TypeError(
                    f"{self.cls.__name__} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. "
                    "You can turn off determinism for this DataPipe if that is acceptable "
                    "for your application"
                )
            return self.cls(*args, **kwargs)

        # Decorate with a functional argument
        if not (isinstance(args[0], type) and issubclass(args[0], IterDataPipe)):
            raise TypeError(
                f"Only `IterDataPipe` can be decorated, but {args[0].__name__} is found"
            )
        self.cls = args[0]
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        res = self.deterministic_fn(*args, **kwargs)
        if not isinstance(res, bool):
            raise TypeError(
                "deterministic_fn of `non_deterministic` decorator is required "
                f"to return a boolean value, but {type(res)} is found"
            )
        global _determinism
        if _determinism and res:
            raise TypeError(
                f"{self.cls.__name__} is non-deterministic with the inputs, but you set "  # type: ignore[union-attr]
                "'guaranteed_datapipes_determinism'. You can turn off determinism "
                "for this DataPipe if that is acceptable for your application"
            )
        return self.cls(*args, **kwargs)  # type: ignore[misc]


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
            if argument_name in hints and isinstance(
                hints[argument_name], _DataPipeMeta
            ):
                hint = hints[argument_name]
                if not isinstance(value, IterDataPipe):
                    raise TypeError(
                        f"Expected argument '{argument_name}' as a IterDataPipe, but found {type(value)}"
                    )
                if not value.type.issubtype(hint.type):
                    raise TypeError(
                        f"Expected type of argument '{argument_name}' as a subtype of "
                        f"hint {hint.type}, but found {value.type}"
                    )

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
    if f.__name__ != "__iter__":
        raise TypeError(
            f"Can not decorate function {f.__name__} with 'runtime_validation'"
        )

    @wraps(f)
    def wrapper(self):
        global _runtime_validation_enabled
        if not _runtime_validation_enabled:
            yield from f(self)
        else:
            it = f(self)
            for d in it:
                if not self.type.issubtype_of_instance(d):
                    raise RuntimeError(
                        f"Expected an instance as subtype of {self.type}, but found {d}({type(d)})"
                    )
                yield d

    return wrapper
