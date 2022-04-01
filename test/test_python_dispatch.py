# Owner(s): ["high priority"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.logging_tensor import LoggingTensor, log_input, capture_logs, no_dispatch
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import enable_python_mode

import logging

class TestPythonDispatch(TestCase):
    def test_basic(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
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
        self.assertRaisesRegexp(
            RuntimeError, "Unable to cast", lambda: A(torch.zeros(1)).detach(),
        )

    def test_detach_appears_twice_when_called_once(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input("x", x)
            x.detach()
        # FIXME: We actually want this to emit a single detach. However,
        # it currently emits two, for reasons unclear to us. Leaving
        # this test here to make sure we don't regress even further (it
        # would be bad if calling .detach() once emits 3+ detaches).
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten.detach($0)
$2 = torch._ops.aten.detach($1)''')

    def test_metadata_change_not_allowed(self) -> None:
        x = LoggingTensor(torch.ones(1))
        y = x.data
        self.assertIsInstance(y, LoggingTensor)
        self.assertRaises(RuntimeError, lambda: y.resize_(4))

    def test_storage(self) -> None:
        # For now, just make sure it doesn't crash.  Ideally, we should
        # return some virtual storage that is safe to work with
        x = LoggingTensor(torch.ones(1))
        self.assertRaises(RuntimeError, lambda: x.storage())

    def test_make_wrapper_subclass_noalloc(self) -> None:
        # This is ludicrously big (8TB) and this should pass because wrapper
        # subclasses don't allocate
        torch.Tensor._make_wrapper_subclass(LoggingTensor, (1000000000000,))

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
            x = LoggingTensor(torch.ones(1), requires_grad=True)
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
        with self.assertRaisesRegex(RuntimeError, err_msg):
            Foo(LoggingTensor(torch.rand(2)))

        with self.assertRaisesRegex(TypeError, "Foo must define __torch_dispatch__"):
            torch.Tensor._make_wrapper_subclass(Foo, (2, 2))

    def test_new_ones(self) -> None:
        class MyTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return MyTensor(3)

        self.assertEqual(type(MyTensor(2).new_ones(3)), MyTensor)

    def test_like(self) -> None:
        class MyTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return MyTensor(3)

        for f in ["empty", "ones", "rand", "randn", "zeros"]:
            f_name = f + "_like"
            self.assertEqual(type(getattr(torch, f_name)(MyTensor(2))), MyTensor)

        self.assertEqual(type(torch.full_like(MyTensor(2), 1.)), MyTensor)
        self.assertEqual(type(torch.randint_like(MyTensor(2), high=3)), MyTensor)

    def test_make_wrapper_subclass_propagates_metadata(self) -> None:
        class WrapperTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls, elem.size(),
                    dtype=elem.dtype, layout=elem.layout,
                    device=elem.device, requires_grad=elem.requires_grad,
                    strides=elem.stride(), storage_offset=elem.storage_offset())
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise RuntimeError("NYI")

        # non-contiguous strides, non-zero storage offset
        x = torch.randn(4, 6).t().diagonal(offset=2)
        y = WrapperTensor(x)
        self.assertEqual(y.size(), x.size())
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.storage_offset(), x.storage_offset())

    def test_index_put_where_only_index_is_subclass(self) -> None:
        called_funcs = []

        class MyTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl
            elem: torch.Tensor
            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(
                    cls, elem.size(),
                    dtype=elem.dtype, layout=elem.layout,
                    device=elem.device, requires_grad=elem.requires_grad
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called_funcs.append(func)
                return MyTensor(torch.tensor(3))

        x = torch.randn(3, 3)
        idxs = (MyTensor(torch.tensor(0)),)
        v = torch.randn(1)
        res = x.index_put_(idxs, v)
        self.assertEqual(called_funcs, [torch.ops.aten.index_put_])

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

    def test_tolist_numpy_with_python_mode(self) -> None:
        x = LoggingTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(RuntimeError, "is not supported for tensor subclasses."):
            x.tolist()
        with self.assertRaisesRegex(RuntimeError, "is not supported for tensor subclasses."):
            x.numpy()
        with self.assertRaises(AssertionError):
            self.assertEqual(x, None)

    def test_enable_python_mode_subclass_autograd_device_check(self) -> None:
        class NonWrapperSubclass(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # Wrong device here!
                r = torch.Tensor._make_subclass(cls, elem.to("meta"), elem.requires_grad)
                # ...the real tensor is held as an element on the tensor.
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                def unwrap(e):
                    return e.elem if isinstance(e, NonWrapperSubclass) else e

                def wrap(e):
                    return NonWrapperSubclass(e) if isinstance(e, torch.Tensor) else e

                # no_dispatch is only needed if you use enable_python_mode.
                # It prevents infinite recursion.
                with no_dispatch():
                    rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
                logging.getLogger("NonWrapperSubclass").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)
                return rs

        x = NonWrapperSubclass(torch.tensor([3.0, 4.0], requires_grad=True))
        y = torch.randn(2, requires_grad=True)
        z = x * y
        self.assertIsInstance(z, NonWrapperSubclass)
        z.sum().backward(torch.tensor(1))
        self.assertEqual(x.grad, y)
        self.assertEqual(y.grad, x)

    def test_none_wrapping(self):
        # A Tensor subclass that returns None when doing add
        # See LoggingTensor above for more details on the subclass
        class SubclassWithNone(torch.Tensor):
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(
                    cls, elem.size(),
                    dtype=elem.dtype, layout=elem.layout,
                    device=elem.device, requires_grad=elem.requires_grad
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                def unwrap(e):
                    return e.elem if isinstance(e, SubclassWithNone) else e

                def wrap(e):
                    return SubclassWithNone(e) if isinstance(e, torch.Tensor) else e

                # no_dispatch is only needed if you use enable_python_mode.
                # It prevents infinite recursion.
                with no_dispatch():
                    rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
                if func.__name__ == "add":
                    return None
                else:
                    return rs

        x = SubclassWithNone(torch.rand(2))
        # Make sure both run without error
        self.assertIsInstance(x * 2, SubclassWithNone)
        self.assertIsNone(x + 2)

        x.requires_grad_()
        out = x.acos().sum()

        # The backward of acos does add then rsqrt so here we make sure that the
        # undefined Tensor generated by the user code is nicely handled.
        # If acos formula changes in the future, this can be replaced by any other
        # function that does add then something in the backward in a composite way
        with self.assertRaisesRegex(RuntimeError, "but got None"):
            out.backward()

    def test_storage_can_be_converted_to_python_object(self):
        with enable_python_mode(LoggingTensor):
            s = torch.Storage()
            z = LoggingTensor(torch.empty([]))
            z.set_(s)

    def test_autograd_in_attr(self):
        # We want the wrapped Tensor to require gradients!
        true_t = torch.rand(2, requires_grad=True)
        t = LoggingTensor(true_t)

        out = t + 2

        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

        self.assertTrue(out.elem.requires_grad)
        self.assertIsNotNone(out.elem.grad_fn)

        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            out.sum().backward()

        out.elem.sum().backward()

        self.assertIsNone(t.grad)
        self.assertIsNotNone(t.elem.grad)


if __name__ == '__main__':
    run_tests()
