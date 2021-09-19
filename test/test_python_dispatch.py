import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import enable_python_mode

from typing import Iterator, List
import logging
import contextlib
import itertools

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

WRAPPER_DEVICE = "meta"
# TODO: TensorBase should work
class LoggingTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (LoggingTensor) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        r = torch.Tensor._make_subclass(cls, elem.to(WRAPPER_DEVICE), elem.requires_grad)
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
            o._shortid = self.next_shortid
            self.next_shortid += 1
        return o._shortid

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
    log_list = []
    handler = LoggingTensorHandler(log_list)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
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
            saved_x = y.grad_fn._saved_self
            grad_y = LoggingTensor(torch.tensor([1.0]))
            log_input("grad_y", grad_y)
            g, = torch.autograd.grad((y,), (x,), (grad_y,))

        self.assertEqual(g.elem, torch.tensor([6.0]))
        with torch.no_grad():
            self.assertEqual(saved_x, x)
            self.assertEqual(saved_x._version, x._version)
            x.add_(2)
            self.assertEqual(saved_x, x)
            # TODO: figure out why broken
            # self.assertEqual(saved_x._version, x._version)
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten.mul($0, $0)
$2 = input('grad_y')
$3 = torch._ops.aten.mul($2, $0)
$4 = torch._ops.aten.mul($2, $0)
$5 = torch._ops.aten.add($4, $3)''')

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
$2 = torch._ops.aten.abs($0, out=$1)''')


    def test_kwarg_only(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.ones(1, 1))
            z = LoggingTensor(torch.ones(1))
            log_input("x", x)
            log_input("y", y)
            log_input("z", z)
            torch.addmv(x, y, z)
            torch.addmv(x, y, z, beta=1)
            torch.addmv(x, y, z, beta=2)
            torch.addmv(x, y, z, alpha=2)
            torch.addmv(x, y, z, beta=2, alpha=2)

        # The expectation is that beta/alpha don't show up when they're
        # defaulted.  This is even if the user explicitly specified it.
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = input('y')
$2 = input('z')
$3 = torch._ops.aten.addmv($0, $1, $2)
$4 = torch._ops.aten.addmv($0, $1, $2)
$5 = torch._ops.aten.addmv($0, $1, $2, beta=2)
$6 = torch._ops.aten.addmv($0, $1, $2, alpha=2)
$7 = torch._ops.aten.addmv($0, $1, $2, beta=2, alpha=2)''')

    def test_kwarg_only_and_positional_default(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.ones(1))
            log_input("x", x)
            log_input("y", y)
            torch.ops.aten.kl_div(x, y)
            torch.ops.aten.kl_div(x, y, 2)
            torch.ops.aten.kl_div(x, y, log_target=True)
            torch.ops.aten.kl_div(x, y, 2, log_target=True)

        # What we are testing here is that we omit reduction
        # if it is defaulted, even if a kwarg is set
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = input('y')
$2 = torch._ops.aten.kl_div($0, $1)
$3 = torch._ops.aten.kl_div($0, $1, 2)
$4 = torch._ops.aten.kl_div($0, $1, log_target=True)
$5 = torch._ops.aten.kl_div($0, $1, 2, log_target=True)''')

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

        # Wobbles depending on NDEBUG mode of pybind11
        self.assertRaisesRegexp(
            RuntimeError, "Unable to cast", lambda: A(torch.zeros(1)).neg(),
        )
        self.assertExpectedRaisesInline(
            RuntimeError, lambda: A(torch.zeros(1)).detach(),
            """detach returned invalid type str, expected Tensor"""
        )

    def test_metadata_change_not_allowed(self) -> None:
        x = LoggingTensor(torch.ones(1))
        y = x.data
        self.assertIsInstance(y, LoggingTensor)
        self.assertRaises(RuntimeError, lambda: y.resize_(4))

    def test_version(self) -> None:
        x = LoggingTensor(torch.ones(1))
        prev_vc = x._version
        x.detach().add_(2)
        cur_vc = x._version
        self.assertNotEqual(prev_vc, cur_vc)
        x.data.add_(2)
        self.assertEqual(cur_vc, x._version)

    def test_subclass_priority(self) -> None:
        class ErrorA(RuntimeError):
            pass

        class ErrorB(RuntimeError):
            pass

        # The big tests for code coverage are test_precedence_semantics in
        # test_overrides.py; this is just to make sure it is wired up at all
        # correctly for __torch_dispatch__
        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise ErrorA

        class B(A):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise ErrorB

        self.assertRaises(ErrorA, lambda: torch.add(A(torch.empty(1)), A(torch.empty(1))))
        self.assertRaises(ErrorB, lambda: torch.add(A(torch.empty(1)), B(torch.empty(1))))
        self.assertRaises(ErrorB, lambda: torch.add(B(torch.empty(1)), A(torch.empty(1))))
        self.assertRaises(ErrorB, lambda: torch.add(B(torch.empty(1)), B(torch.empty(1))))

    def test_format(self) -> None:
        x = LoggingTensor(torch.ones(1))
        s1 = str(x)
        s2 = repr(x)
        s3 = f"{x}"
        self.assertExpectedInline(s1, """LoggingTensor(tensor([1.]))""")
        self.assertEqual(s1, s2)
        self.assertEqual(s1, s3)

    def test_custom_autograd(self) -> None:
        escape = [None]

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x ** 2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                assert isinstance(grad_output, LoggingTensor)
                x, = ctx.saved_tensors
                assert isinstance(x, LoggingTensor)
                escape[0] = x
                return grad_output * 2 * x

        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1, requires_grad=True))
            log_input("x", x)
            x.grad = LoggingTensor(torch.zeros(1))
            log_input("x.grad", x.grad)
            y = Square.apply(x)
            grad_output = LoggingTensor(torch.ones(1))
            log_input("grad_output", grad_output)
            y.backward(grad_output)

        with torch.no_grad():
            self.assertEqual(escape[0], x)
            self.assertEqual(escape[0]._version, x._version)
            # TODO: figure out why x.requires_grad = False doesn't
            # trigger an error for LoggingTensor
            x.add_(2)
            self.assertEqual(escape[0], x)
            # TODO: figure out why this is broken
            # self.assertEqual(escape[0]._version, x._version)

        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = input('x.grad')
$2 = torch._ops.aten.pow($0, 2)
$3 = input('grad_output')
$4 = torch._ops.aten.mul($3, tensor(2))
$5 = torch._ops.aten.mul($4, $0)
$6 = torch._ops.aten.add_($1, $5)''')

    def test_subclass_creation(self):
        # Make sure these statements runs without error
        # In particular checking that when internal detach returns
        # subclasses, these are cleanly overwritten.
        class Foo(torch.Tensor):
            pass

        err_msg = "subclass Foo but.*already associated to a python object of type LoggingTensor"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            a = torch.Tensor._make_subclass(Foo, LoggingTensor(torch.rand(2)))
        with self.assertRaisesRegex(RuntimeError, err_msg):
            b = LoggingTensor(torch.rand(2)).as_subclass(Foo)

        # And in case where we don't know if the user wants this subclass
        # overwritten, raise a nice error.
        # The standard LoggingTensor will fail because it is not on the right device
        with self.assertRaisesRegex(TypeError, "expected.*device=cpu.*device=meta"):
            Foo(LoggingTensor(torch.rand(2)))

        # And if we put it on the right device, we still get a nice error
        try:
            global WRAPPER_DEVICE
            prev_device = WRAPPER_DEVICE
            WRAPPER_DEVICE = "cpu"

            err_msg = "Creating a new Tensor subclass Foo.*python object of type LoggingTensor"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                Foo(LoggingTensor(torch.rand(2)))

        finally:
            WRAPPER_DEVICE = prev_device

    def test_enable_python_mode_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "__torch_dispatch__"):
            with enable_python_mode(torch.Tensor):
                pass
        z = LoggingTensor(torch.empty([]))
        with self.assertRaisesRegex(ValueError, "must be the type"):
            with enable_python_mode(z):
                pass

    def test_enable_python_mode_basic(self) -> None:
        with enable_python_mode(LoggingTensor):
            z = torch.empty([])
            self.assertTrue(isinstance(z, LoggingTensor))

    def test_enable_python_mode_unrelated_tensors(self) -> None:
        x = torch.randn([])
        y = torch.randn([])
        with enable_python_mode(LoggingTensor):
            z = x + y
            self.assertTrue(isinstance(z, LoggingTensor))

    def test_enable_python_mode_subclass_priority(self) -> None:
        class ErrorA(RuntimeError):
            pass

        class ErrorB(RuntimeError):
            pass

        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise ErrorA

        class B(A):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise ErrorB

        a = A(torch.empty(1))
        b = B(torch.empty(1))
        with self.assertRaises(ErrorA):
            a + a

        # B has precedence over A due to the subclass relationship
        with self.assertRaises(ErrorB):
            with enable_python_mode(A):
                b + b
        with self.assertRaises(ErrorB):
            with enable_python_mode(B):
                a + a
        with self.assertRaises(ErrorB):
            with enable_python_mode(B):
                a + b

    def test_enable_python_mode_respects_no_dispatch(self) -> None:
        with enable_python_mode(LoggingTensor):
            z = torch.ones([2, 3])
            self.assertTrue(isinstance(z, LoggingTensor))
            with no_dispatch():
                expected = torch.ones([2, 3])
                self.assertEqual(z.elem, expected)

    def test_nested_enable_python_mode(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            with enable_python_mode(LoggingTensor):
                with enable_python_mode(LoggingTensor):
                    pass

if __name__ == '__main__':
    run_tests()
