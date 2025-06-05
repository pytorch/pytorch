import types
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Callable


# The idea for this parameter is that we forbid bare assignment
# to torch.backends.<cudnn|mkldnn>.enabled and friends when running our
# test suite, where it's very easy to forget to undo the change
# later.
__allow_nonbracketed_mutation_flag = True


def disable_global_flags() -> None:
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False


def flags_frozen() -> bool:
    return not __allow_nonbracketed_mutation_flag


@contextmanager
def __allow_nonbracketed_mutation() -> Generator[None, None, None]:
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


class ContextProp:
    def __init__(
        self, getter: Callable[[], Any], setter: Callable[[Any], None]
    ) -> None:
        self.getter = getter
        self.setter = setter

    def __get__(self, obj: Any, objtype: Any) -> Any:
        return self.getter()

    def __set__(self, obj: Any, val: Any) -> None:
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context manager instead"
            )


class PropModule(types.ModuleType):
    def __init__(self, m: types.ModuleType, name: str) -> None:
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr: str) -> Any:
        return self.m.__getattribute__(attr)


from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    cusparselt as cusparselt,
    kleidiai as kleidiai,
    mha as mha,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    quantized as quantized,
)
