import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._C import _disabled_torch_function_impl
from torch.utils._pytree import tree_map

from typing import Iterator, List
import logging
import contextlib

# TODO: move this into library proper
@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
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
        # The wrapping tensor (LoggingTensor) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def __repr__(self):
        return f"LoggingTensor({self.elem})"

    def __str__(self):
        return f"LoggingTensor({self.elem})"

    def __format__(self, format_spec):
        return f"LoggingTensor({self.elem})"

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, torch.Tensor) else e

        def wrap(e):
            return LoggingTensor(e) if isinstance(e, torch.Tensor) else e

        # TODO: handle kwargs
        assert not kwargs
        rs = tree_map(wrap, func(*tree_map(unwrap, args)))
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, rs)
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
            o._shortid = self.next_shortid
            self.next_shortid += 1
        return o._shortid

    def _fmt(self, a: object) -> str:
        return f'${self._shortid(a)}' if isinstance(a, LoggingTensor) else repr(a)

    def emit(self, record):
        fmt_args = "(" + ", ".join(self._fmt(a) for a in record.args[0]) + ")"
        fmt_rets = ", ".join(self._fmt(a) for a in record.args[1]) \
            if isinstance(record.args[1], (list, tuple)) else self._fmt(record.args[1])
        self.log_list.append(f'{fmt_rets} = {record.msg}{fmt_args}')

def log_input(name: str, var: object):
    logging.getLogger("LoggingTensor").info("input", (name,), (var,))

@contextlib.contextmanager
def capture_logs() -> Iterator[List[str]]:
    logger = logging.getLogger("LoggingTensor")
    log_list = []
    handler = LoggingTensorHandler(log_list)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield log_list
    finally:
        logger.removeHandler(handler)

class TestPythonDispatch(TestCase):
    def test_basic(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0], requires_grad=True))
            log_input("x", x)
            y = x * x
            grad_y = LoggingTensor(torch.tensor([1.0]))
            log_input("grad_y", grad_y)
            g, = torch.autograd.grad((y,), (x,), (grad_y,))

        self.assertEqual(g.elem, torch.tensor([6.0]))
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten.detach($0)
$2 = torch._ops.aten.detach($0)
$3 = torch._ops.aten.mul($0, $0)
$4 = input('grad_y')
$5 = torch._ops.aten.detach($1)
$6 = torch._ops.aten.detach($2)
$7 = torch._ops.aten.mul($4, $5)
$8 = torch._ops.aten.mul($4, $6)
$9 = torch._ops.aten.add($8, $7, 1)''')

    def test_out(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.zeros(1))
            log_input("x", x)
            log_input("y", y)
            torch.abs(x, out=y)

        self.assertEqual(y.elem, torch.ones(1))
        # TODO: arguably this shouldn't pass and we should complain
        # that out isn't a kwarg
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = input('y')
$2 = torch._ops.aten.abs($0, $1)''')

    def test_list_ret(self) -> None:
        # test all sequence types are permissible returns
        for list_type in (list, tuple):
            class A(torch._C._TensorBase):
                @staticmethod
                def __new__(cls, elem):
                    return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    if func == torch.ops.aten.split:
                        with no_dispatch():
                            return list_type(torch.split(*args))
                    else:
                        raise AssertionError(f"unrecognized func: {func}")

            self.assertEqual(
                torch.split(A(torch.tensor([0, 1])), 2),
                torch.split(torch.tensor([0, 1]), 2)
            )

    def test_invalid_ret(self) -> None:
        # test invalid return gets reasonable error message
        class A(torch._C._TensorBase):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return "arf"

        self.assertExpectedRaisesInline(
            RuntimeError, lambda: A(torch.zeros(1)).neg(),
            """Unable to cast Python instance of type <class 'str'> to C++ type 'at::Tensor'"""
        )
        self.assertExpectedRaisesInline(
            RuntimeError, lambda: A(torch.zeros(1)).detach(),
            """detach returned invalid type str, expected Tensor"""
        )


if __name__ == '__main__':
    run_tests()
