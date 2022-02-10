import torch
from torch.utils._pytree import tree_map

from typing import Iterator, List
import logging
import contextlib
import itertools

# TODO: move this into library proper
@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


# How the chain of calls works for LoggingTensor:
# 1. Call torch.sin
# 2. Attempt __torch_function__. In LoggingTensor torch function is disabled so we bypass it entirely
# 3. Enter dispatcher, wind your way through Autograd
# 4. Hit Python dispatch key, call __torch_dispatch__

# TODO: TensorBase should work
class LoggingTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (LoggingTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def __repr__(self):
        return f"LoggingTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, LoggingTensor) else e

        def wrap(e):
            return LoggingTensor(e) if isinstance(e, torch.Tensor) else e

        # no_dispatch is only needed if you use enable_python_mode.
        # It prevents infinite recursion.
        with no_dispatch():
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)
        return rs

# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
class LoggingTensorHandler(logging.Handler):
    log_list: List[str]
    next_shortid: int

    def __init__(self, log_list: List[str]) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.next_shortid = 0

    # WARNING: not deterministic over multiple threads, this matters for
    # autograd
    def _shortid(self, o: object) -> int:
        if not hasattr(o, '_shortid'):
            o._shortid = self.next_shortid  # type: ignore[attr-defined]
            self.next_shortid += 1
        return o._shortid  # type: ignore[attr-defined]

    def _fmt(self, a: object) -> str:
        return f'${self._shortid(a)}' if isinstance(a, LoggingTensor) else repr(a)

    def emit(self, record):
        fmt_args = ", ".join(itertools.chain(
            (self._fmt(a) for a in record.args[0]),
            (f"{k}={self._fmt(v)}" for k, v in record.args[1].items())
        ))
        fmt_rets = ", ".join(self._fmt(a) for a in record.args[2]) \
            if isinstance(record.args[2], (list, tuple)) else self._fmt(record.args[2])
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')

def log_input(name: str, var: object):
    logging.getLogger("LoggingTensor").info("input", (name,), {}, (var,))

@contextlib.contextmanager
def capture_logs() -> Iterator[List[str]]:
    logger = logging.getLogger("LoggingTensor")
    log_list: List[str] = []
    handler = LoggingTensorHandler(log_list)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield log_list
    finally:
        logger.removeHandler(handler)
