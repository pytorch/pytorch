from typing import Any, Callable, Optional, Type, Union
from torch.utils.data import IterDataPipe


class functional_datapipe(object):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, cls):
        if not (isinstance(cls, Callable) or issubclass(cls, IterDataPipe)):
            raise Exception('Can only decorate IterDataPipe')
        IterDataPipe.register_datapipe_as_function(self.name, cls)
        return cls


_determinism: bool = False


class guaranteed_datapipes_determinism(object):
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


class non_deterministic(object):
    cls: Optional[Type[IterDataPipe]] = None
    deterministic_fn: Callable[[], bool]

    def __init__(self, arg: Union[Type[IterDataPipe], Callable[[], bool]]) -> None:
        # 1. Decorator doesn't have any argument
        if isinstance(arg, Type):
            if not issubclass(arg, IterDataPipe):
                raise TypeError("only `IterDataPipe` can be decorated, but {} is found"
                                .format(arg.__name__))
            self.cls = arg
        # 2. Decorator has an argument of a function
        #    This class should behave differently given different inputs. Use this
        #    function to verify the determinism for each instance.
        #    When the function returns True, the instance is non-deterministic. Otherwise,
        #    the instance is a deterministic DataPipe.
        elif isinstance(arg, Callable):
            self.deterministic_fn = arg
        else:
            raise TypeError("{} can not be decorated by non_deterministic".format(arg))

    def __call__(self, *args, **kwargs):
        global _determinism
        if _determinism and self.cls is not None:
            raise TypeError("{} is non-deterministic, but you set 'guaranteed_datapipes_determinism'. "
                            "You can turn off determinism for this DataPipe if that is acceptable "
                            "for your application".format(self.cls.__name__))

        if self.cls is not None:
            return self.cls(*args, **kwargs)

        if not issubclass(args[0], IterDataPipe):
            raise TypeError("only `IterDataPipe` can be decorated, but {} is found"
                            .format(args[0].__name__))
        self.cls = args[0]
        return self.deterministic_wrapper_fn

    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe:
        res = self.deterministic_fn(*args, **kwargs)
        if not isinstance(res, bool):
            raise TypeError("deterministic_fn of `non_deterministic` decorator is required "
                            "to return boolean value, but {} is found".format(type(res)))
        global _determinism
        if _determinism and res:
            raise TypeError("{} is non-deterministic with the inputs, but you set "
                            "'guaranteed_datapipes_determinism'. You can turn off determinism "
                            "for this DataPipe if that is acceptable for your application"
                            .format(self.cls.__name__))
        return self.cls(*args, **kwargs)
