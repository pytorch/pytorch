import torch
from torch.utils._python_dispatch import TorchDispatchMode
import logging
import itertools

from typing import List

class PrintingMode(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    print(f"{func.__module__}.{func.__name__}({args}, {kwargs})")
    return func(*args, **kwargs)

class LoggingTensorHandler(logging.Handler):
    log_list: List[str]
    next_shortid: int

    def __init__(self, log_list: List[str], use_shortid_for_all_tensors: bool) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.next_shortid = 0
        self.use_shortid_for_all_tensors = use_shortid_for_all_tensors

    # WARNING: not deterministic over multiple threads, this matters for
    # autograd
    def _shortid(self, o: object) -> int:
        if not hasattr(o, '_shortid'):
            o._shortid = self.next_shortid  # type: ignore[attr-defined]
            self.next_shortid += 1
        return o._shortid  # type: ignore[attr-defined]

    def _fmt(self, a: object) -> str:
        cond_cls = torch.Tensor if self.use_shortid_for_all_tensors else LoggingTensor
        return f'${self._shortid(a)}' if isinstance(a, cond_cls) else repr(a)

    def emit(self, record):
        fmt_args = ", ".join(itertools.chain(
            (self._fmt(a) for a in record.args[0]),
            (f"{k}={self._fmt(v)}" for k, v in record.args[1].items())
        ))
        fmt_rets = ", ".join(self._fmt(a) for a in record.args[2]) \
            if isinstance(record.args[2], (list, tuple)) else self._fmt(record.args[2])
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')

logger = logging.getLogger("test")
logs: List[str] = []
logger.addHandler(LoggingTensorHandler(logs, True))
logger.setLevel(logging.INFO)
class LoggingMode(TorchDispatchMode):
  def __init__(self, logger):
    self.logger = logger
    return super().__init__()

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    res = func(*args, **kwargs)
    self.logger.info(f"{func.__module__}.{func.__name__}", args, kwargs, res)
    return res

with LoggingMode(logger):
  x = torch.randn(3, 4)
  y = x + x

breakpoint()
print(y)

