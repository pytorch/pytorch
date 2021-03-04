from typing import Any, Type
from torch.utils.data import IterDataPipe


class functional_datapipe(object):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, cls):
        if not (isinstance(cls, non_deterministic) or issubclass(cls, IterDataPipe)):
            raise Exception('Can only decorate IterDataPipe')
        IterDataPipe.register_datapipe_as_function(self.name, cls)
        return cls


_determinism: bool = False


class set_determinism(object):
    prev: bool

    def __init__(self, mode: bool) -> None:
        assert isinstance(mode, bool), "set_determinism expects a bool, but got {}".format(type(mode))
        global _determinism
        self.prev = _determinism
        _determinism = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _determinism
        _determinism = self.prev


class non_deterministic(object):
    cls: Type[IterDataPipe]

    def __init__(self, cls: Type[IterDataPipe]) -> None:
        if not issubclass(cls, IterDataPipe):
            raise TypeError("Only `IterDataPipe` can be decorated, but {} is found".format(cls.__name__))
        self.cls = cls

    def __call__(self, *args, **kwargs):
        global _determinism
        if _determinism:
            raise TypeError("{} is non-deterministic DataPipe, but you set 'set_determinism(True)' "
                            "You can turn off determinism for this DataPipe if that is acceptable "
                            "for your application".format(self.cls.__name__))
        return self.cls(*args, **kwargs)
