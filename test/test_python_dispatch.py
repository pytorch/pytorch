# Owner(s): ["module: __torch_dispatch__"]

import tempfile
import torch
from copy import deepcopy
from torch.library import Library
from torch.cuda.jiterator import _create_jit_fn
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, IS_WINDOWS
from torch.utils._mode_utils import no_dispatch, all_same_mode
from torch.testing._internal.logging_tensor import LoggingTensor, LoggingTensorReentrant, LoggingTensorMode, \
    log_input, capture_logs, capture_logs_with_logging_tensor_mode
from torch.utils._pytree import tree_map, tree_map_only
from torch.utils._python_dispatch import enable_torch_dispatch_mode, TorchDispatchMode

import logging


class TestPythonRegistration(TestCase):
    def test_override_aten_ops_with_multiple_libraries(self) -> None:
        x = torch.tensor([1, 2])
        my_lib1 = Library("aten", "IMPL")
        my_lib2 = Library("aten", "IMPL")

        # Example 1
        def my_neg(*args, **kwargs):
            return args[0]._neg_view()

        # Now we are secretly making the operator a view op so autograd needs to know how
        # to handle it
        my_lib1.impl('neg', my_neg, "AutogradCPU")

        self.assertTrue(torch.neg(x).is_neg())

        # RuntimeError: impl("aten::neg", ...):
        # Explicitly provided namespace (aten) in operator name does not match ...
        with self.assertRaisesRegex(RuntimeError, "operator name does not match namespace"):
            my_lib3 = Library("foo", "DEF")
            my_lib3.define("neg(Tensor self) -> Tensor")
            my_lib3.impl(torch.ops.aten.neg.default, my_neg, "AutogradCPU")
            del my_lib3

        # Example 2
        def my_mul(*args, **kwargs):
            return torch.zeros_like(args[0])

        # torch.ops.aten.mul.Tensor
        my_lib2.impl("aten::mul.Tensor", my_mul, "ZeroTensor")

        y = torch._efficientzerotensor(2)
        self.assertFalse(torch.mul(x, y)._is_zerotensor())

        # Assert that a user can't override the behavior of a (ns, op, dispatch_key)
        # combination if someone overrided the behavior for the same before them
        with self.assertRaisesRegex(RuntimeError, 'already a kernel registered from python'):
            my_lib2.impl(torch.ops.aten.mul.Tensor, my_mul, "ZeroTensor")

        del my_lib1

        # Validate that lib2 is not affected by removing lib1
        self.assertFalse(torch.mul(x, y)._is_zerotensor())

        del my_lib2

        # Validate that the old behavior is restored for neg and mul
        self.assertFalse(torch.neg(x).is_neg())
        self.assertTrue(torch.mul(x, y)._is_zerotensor())

    def test_error_if_fn_not_callable(self):
        with self.assertRaisesRegex(TypeError, "Input function is required to be a callable"):
            my_lib = Library("aten", "IMPL")
            my_lib.impl(torch.ops.aten.neg.default, [], "AutogradCPU")

    def test_override_cpu_sum(self) -> None:
        # Example 1
        run = [False]

        def my_sum(*args, **kwargs):
            run[0] = True
            return args[0]

        my_lib1 = Library("aten", "IMPL")
        my_lib1.impl('aten::sum', my_sum, "CPU")
        x = torch.tensor([1, 2])
        self.assertEqual(torch.sum(x), x)
        self.assertTrue(run[0])
        del my_lib1
        # Validate that the old behavior is restored for sum
        self.assertEqual(torch.sum(x), torch.tensor(3))

    def test_override_cuda_with_jiterator(self) -> None:
        def override_where_cuda() -> None:
            # Example 1: Invert the behavior of where's condition input
            not_where_code_string = '''
            template <typename T> T inverted_where(bool cond, T a, T b){
                return !cond ? a : b;
            }
            '''
            jitted_where = _create_jit_fn(not_where_code_string)

            CALLED = [False]

            def inverted_where(*args, **kwargs):
                CALLED[0] = True
                return jitted_where(*args, **kwargs)

            # overriding where's cuda kernel with Jiterator generated kernel
            my_lib = Library("aten", "IMPL")
            my_lib.impl('aten::where.self', inverted_where, "CUDA")

            device = 'cuda'
            cond = torch.tensor([True, True, False], device=device, dtype=torch.bool)
            x = torch.tensor([1, 2, 3], device=device)
            y = torch.tensor([-1, -2, -3], device=device)

            self.assertEqual(torch.where(cond, x, y), torch.tensor([-1, -2, 3]))
            self.assertTrue(CALLED[0])
            del my_lib

            # behavior restored after deregistration
            self.assertEqual(torch.where(cond, x, y), torch.tensor([1, 2, -3]))

        def override_gelu_cuda() -> None:
            # Example 2: Use relu to approximate gelu for faster compute
            fastest_gelu_code_string = '''
            template <typename T> T fast_gelu(T a){
                return a > 0 ? a : 0;
            }
            '''
            jitted_gelu = _create_jit_fn(fastest_gelu_code_string)

            CALLED = [False]

            def fast_gelu(*args, **kwargs):
                CALLED[0] = True
                return jitted_gelu(*args, **kwargs)

            # overriding gelu's cuda kernel with Jiterator generated relu kernel
            my_lib = Library("aten", "IMPL")
            my_lib.impl('aten::gelu', fast_gelu, "CUDA")

            x = torch.rand([3, 3], device='cuda', dtype=torch.float)
            self.assertEqual(torch.nn.functional.gelu(x), torch.nn.functional.relu(x))
            self.assertTrue(CALLED[0])
            del my_lib

            # behavior restored after deregistration
            self.assertNotEqual(torch.nn.functional.gelu(x), torch.nn.functional.relu(x))

        def override_exp_cuda() -> None:
            # Example 3: Preventing exp from exploding for float16
            clipped_exp_code_string = '''
            template <typename T> T clipped_exp(T a){
                return a > T(10.0) ? T(22026.4657948) : exp(a);
            }
            '''
            jitted_exp = _create_jit_fn(clipped_exp_code_string)

            CALLED = [False]

            def clipped_exp(*args, **kwargs):
                CALLED[0] = True
                return jitted_exp(*args, **kwargs)

            # overriding exp's cuda kernel with clipped_exp kernel
            my_lib = Library("aten", "IMPL")
            my_lib.impl('aten::exp', clipped_exp, "CUDA")

            x = torch.tensor([0.0, 100.0], device='cuda', dtype=torch.float16)
            self.assertEqual(torch.exp(x), torch.tensor([1.0, 22026.4657948], dtype=torch.float16))
            self.assertTrue(CALLED[0])
            del my_lib

            # behavior restored after deregistration
            self.assertEqual(torch.exp(x), torch.tensor([1.0, torch.inf], dtype=torch.float16))

        def override_add_cuda() -> None:
            # Example 4: simulate a hardware bug, where the adder is always off by 1
            buggy_add_code_string = '''
            template <typename T> T buggy_add(T a, T b){
                return a + b + T(1);
            }
            '''
            jitted_add = _create_jit_fn(buggy_add_code_string)

            CALLED = [False]

            def buggy_add(*args, **kwargs):
                CALLED[0] = True
                return jitted_add(*args, **kwargs)

            my_lib = Library("aten", "IMPL")
            my_lib.impl('aten::add.Tensor', buggy_add, "CUDA")

            x_cpu = torch.rand([3, 3], device='cpu')
            y_cpu = torch.rand([3], device='cpu')

            x_cuda = x_cpu.cuda()
            y_cuda = y_cpu.cuda()

            self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu + 1)
            self.assertTrue(CALLED[0])
            del my_lib

            # behavior restored after deregistration
            self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu)

        if torch.cuda.is_available() and not TEST_WITH_ROCM:
            override_where_cuda()
            override_gelu_cuda()
            override_exp_cuda()
            override_add_cuda()

    def test_extend_library_with_dispatch_key_arg(self):
        def my_sum(*args, **kwargs):
            return args[0]
        my_lib1 = Library("aten", "IMPL", dispatch_key="CPU")

        # RuntimeError: Explicitly provided dispatch key (Conjugate) is
        # inconsistent with the dispatch key of the enclosing TORCH_LIBRARY_IMPL block
        with self.assertRaisesRegex(RuntimeError, "inconsistent with the dispatch key"):
            my_lib1.impl('sum', my_sum, "Conjugate")
        my_lib1.impl('aten::sum', my_sum)
        x = torch.tensor([1, 2])
        self.assertEqual(torch.sum(x), x)
        del my_lib1

    def test_create_new_library(self) -> None:
        my_lib1 = Library("foo", "DEF")

        my_lib1.define("sum(Tensor self) -> Tensor")

        # Example 1
        @torch.library.impl(my_lib1, "sum", "CPU")
        def my_sum(*args, **kwargs):
            return args[0]

        x = torch.tensor([1, 2])
        self.assertEqual(torch.ops.foo.sum(x), x)

        my_lib2 = Library("foo", "IMPL")

        # Example 2
        @torch.library.impl(my_lib2, torch.ops.foo.sum.default, "ZeroTensor")
        def my_sum_zt(*args, **kwargs):
            if args[0]._is_zerotensor():
                return torch._efficientzerotensor(args[0].shape)
            else:
                return args[0]

        y = torch._efficientzerotensor(3)
        self.assertTrue(torch.ops.foo.sum(y)._is_zerotensor())
        self.assertEqual(torch.ops.foo.sum(x), x)

        del my_lib2
        del my_lib1

    @unittest.skipIf(IS_WINDOWS, "Skipped under Windows")
    def test_alias_analysis(self):
        def test_helper(alias_analysis=""):
            my_lib1 = Library("foo", "DEF")

            called = [0]

            @torch.library.define(my_lib1, "_op() -> None", alias_analysis=alias_analysis)
            def _op(*args, **kwargs):
                called[0] += 1

            @torch.jit.script
            def _test():
                torch.ops.foo._op()

            assert "foo::_op" in str(_test.graph)

        with self.assertRaises(AssertionError):
            test_helper("")  # alias_analysis="FROM_SCHEMA"

        test_helper("CONSERVATIVE")

    def test_error_for_unsupported_ns_or_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported kind"):
            my_lib1 = Library("myns", "BLA")

        with self.assertRaisesRegex(ValueError, "reserved namespace"):
            my_lib1 = Library("prim", "DEF")

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
$1 = torch._ops.aten.mul.Tensor($0, $0)
$2 = input('grad_y')
True = torch._ops.aten.is_same_size.default($1, $2)
$3 = torch._ops.aten.mul.Tensor($2, $0)
$4 = torch._ops.aten.mul.Tensor($2, $0)
$5 = torch._ops.aten.add.Tensor($4, $3)''')

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
$2 = torch._ops.aten.abs.out($0, out=$1)''')

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
$3 = torch._ops.aten.addmv.default($0, $1, $2)
$4 = torch._ops.aten.addmv.default($0, $1, $2)
$5 = torch._ops.aten.addmv.default($0, $1, $2, beta=2)
$6 = torch._ops.aten.addmv.default($0, $1, $2, alpha=2)
$7 = torch._ops.aten.addmv.default($0, $1, $2, beta=2, alpha=2)''')

    def test_kwarg_only_and_positional_default(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            log_input("x", x)
            torch.ops.aten._foobar(x)
            torch.ops.aten._foobar(x, False)
            torch.ops.aten._foobar(x, arg3=False)
            torch.ops.aten._foobar(x, False, arg3=False)

        # What we are testing here is that we omit arg2
        # if it is defaulted, even if a kwarg is set
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten._foobar.default($0)
$2 = torch._ops.aten._foobar.default($0, False)
$3 = torch._ops.aten._foobar.default($0, arg3=False)
$4 = torch._ops.aten._foobar.default($0, False, arg3=False)''')

    def test_produce_real_type(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(2, 2))
            log_input("x", x)
            x.to(dtype=torch.double)  # non-optional dtype
            torch.cumprod(x, 0, dtype=torch.double)  # optional dtype
            x[:, 1].contiguous(memory_format=torch.contiguous_format)  # optional memory format
            # There doesn't appear to be any layout signatures which are
            # triggerable using tensor subclasses (need to use a mode)

        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten._to_copy.default($0, dtype=torch.float64)
$2 = torch._ops.aten.cumprod.default($0, 0, dtype=torch.float64)
$3 = torch._ops.aten.slice.Tensor($0, 0, 0, 9223372036854775807)
$4 = torch._ops.aten.select.int($3, 1, 1)
$5 = torch._ops.aten.clone.default($4, memory_format=torch.contiguous_format)''')

    def test_list_ret(self) -> None:
        # test all sequence types are permissible returns
        for list_type in (list, tuple):
            class A(torch._C._TensorBase):
                @staticmethod
                def __new__(cls, elem):
                    return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    if func.overloadpacket == torch.ops.aten.split:
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
        self.assertRaisesRegex(
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
$1 = torch._ops.aten.detach.default($0)
$2 = torch._ops.aten.detach.default($1)''')

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
$2 = torch._ops.aten.pow.Tensor_Scalar($0, 2)
$3 = input('grad_output')
True = torch._ops.aten.is_same_size.default($2, $3)
$4 = torch._ops.aten.mul.Tensor($3, 2)
$5 = torch._ops.aten.mul.Tensor($4, $0)
$6 = torch._ops.aten.add_.Tensor($1, $5)''')

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

    def test_wrapper_subclass_serializes(self) -> None:
        with tempfile.TemporaryFile() as f:
            x = LoggingTensor(torch.randn(3))
            torch.save(x, f)
            f.seek(0)
            x_loaded = torch.load(f)
            self.assertTrue(type(x_loaded) is type(x))
            self.assertEqual(x.elem, x_loaded.elem)
            self.assertFalse(x is x_loaded)

    def test_deepcopy_wrapper_subclass(self) -> None:
        x = LoggingTensor(torch.randn(3))
        x_copy = deepcopy(x)
        self.assertTrue(type(x_copy) is type(x))
        self.assertEqual(x.elem, x_copy.elem)
        self.assertFalse(x is x_copy)

    def test_deepcopy_wrapper_subclass_with_clone_returning_different_type(self) -> None:

        class MyWrapperTensor(torch.Tensor):
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
                if func.overloadpacket.__name__ == "clone":
                    # Return a plain tensor from clone().
                    return args[0].elem.clone()
                raise RuntimeError("NYI")

            # NB: The default Tensor.__torch_function__ implementation called for deepcopy
            # disables __torch_function__ by the time we get to clone(), so there is no need to
            # explicitly disable __torch_function__ for this subclass.

        x = MyWrapperTensor(torch.randn(3))
        with self.assertRaisesRegex(RuntimeError,
                                    "for which cloning returns another instance of the same subclass"):
            x_copy = deepcopy(x)

    def test_deepcopy_non_wrapper_subclass(self) -> None:

        # Ensure correct error is thrown for common error cases.
        class SubTensorError1(torch.Tensor):
            # Default implementation of new_empty() returns a plain tensor.
            pass

        class SubTensorError2(torch.Tensor):
            # new_empty() incorrectly returns a different type (i.e. a plain tensor).
            def new_empty(self, shape):
                return torch.Tensor(shape)

        for error_cls in [SubTensorError1, SubTensorError2]:
            x = error_cls(3)
            with self.assertRaisesRegex(RuntimeError,
                                        "for which that function returns another instance of the same subclass"):
                x_copy = deepcopy(x)

        # Ensure a correctly implemented new_empty() causes deepcopy() to work.
        class SubTensorSuccess(torch.Tensor):
            def new_empty(self, shape):
                return type(self)(shape)

        x = SubTensorSuccess(3)
        x_copy = deepcopy(x)
        self.assertIs(type(x_copy), type(x))

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
        self.assertEqual(called_funcs, [torch.ops.aten.index_put_.default])

    def test_enable_torch_dispatch_mode_error(self) -> None:
        z = LoggingTensor(torch.empty([]))
        with self.assertRaisesRegex(ValueError, "expected to get TorchDispatchMode, Tensor-like class, or None"):
            with enable_torch_dispatch_mode(z):
                pass

    def test_enable_torch_dispatch_mode_basic(self) -> None:
        with capture_logs(is_mode=True) as logs:
            with enable_torch_dispatch_mode(LoggingTensorMode()):
                torch.empty([])
        self.assertExpectedInline('\n'.join(logs), """\
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)""")

    def test_enable_torch_dispatch_mode_unrelated_tensors(self) -> None:
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with enable_torch_dispatch_mode(LoggingTensorMode()):
                x + y
        self.assertExpectedInline('\n'.join(logs), """\
$2 = torch._ops.aten.add.Tensor($0, $1)""")

    def test_nested_push_regular(self):
        with LoggingTensorMode.push() as mode:
            # This previously errored
            with LoggingTensorMode():
                pass

    def test_nested_push_logging_tensor_mode(self):
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                with LoggingTensorMode():
                    torch.empty([])
                    x + y

        self.assertExpectedInline('\n'.join(logs), """\
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3 = torch._ops.aten.add.Tensor($1, $2)
$3 = torch._ops.aten.add.Tensor($1, $2)""")

    def test_capture_logs_with_torch_dispatch_mode(self):
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs_with_logging_tensor_mode() as logs:
            torch.empty([])
            x + y
        self.assertExpectedInline('\n'.join(logs), """\
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3 = torch._ops.aten.add.Tensor($1, $2)""")

        x = torch.randn([])
        y = torch.randn([])

        with capture_logs_with_logging_tensor_mode() as logs1:
            with capture_logs_with_logging_tensor_mode() as logs2:
                torch.empty([])
                x + y

        self.assertExpectedInline('\n'.join(logs2), """\
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3 = torch._ops.aten.add.Tensor($1, $2)
$3 = torch._ops.aten.add.Tensor($1, $2)""")

        self.assertEqual(logs1, logs2)

    def test_enable_torch_dispatch_mode_subclass_priority(self) -> None:
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
        with self.assertRaises(ErrorB):
            a + b

        # B has precedence over A due to the subclass relationship yet
        # modes take precedence over arguments
        with self.assertRaises(ErrorA):
            with enable_torch_dispatch_mode(A):
                b + b
        with self.assertRaises(ErrorB):
            with enable_torch_dispatch_mode(B):
                a + a
        with self.assertRaises(ErrorB):
            with enable_torch_dispatch_mode(B):
                a + b

    def test_enable_torch_dispatch_mode_respects_no_dispatch(self) -> None:
        with capture_logs(is_mode=True) as logs1:
            with enable_torch_dispatch_mode(LoggingTensorMode()):
                torch.ones([2, 3])
                with no_dispatch():
                    torch.ones([2, 3])
        with capture_logs(is_mode=True) as logs2:
            with enable_torch_dispatch_mode(LoggingTensorMode()):
                torch.ones([2, 3])
        self.assertEqual(logs1, logs2)

    def test_enable_torch_dispatch_mode_instance(self) -> None:
        class TestMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        x = TestMode()
        y = torch.tensor([2.])
        with enable_torch_dispatch_mode(x):
            y + y

    def test_shallow_copy_and_detach(self) -> None:
        seen = set()
        test_case = self

        class TestMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                tree_map_only(torch.Tensor, lambda t: test_case.assertIn(t, seen), (args, kwargs))
                if kwargs is None:
                    kwargs = {}
                r = func(*args, **kwargs)
                tree_map_only(torch.Tensor, lambda t: seen.add(t), r)
                return r

        with TestMode():
            x = torch.randn(3, requires_grad=True)
            loss = (x * x).sum()
            loss.backward()

    def test_nested_enable_torch_dispatch_mode(self) -> None:
        class A(LoggingTensorMode):
            pass

        with self.assertRaisesRegex(ValueError, "there is already an active mode"):
            with enable_torch_dispatch_mode(LoggingTensorMode()):
                with enable_torch_dispatch_mode(A()):
                    pass

        # For nesting to be a noop, they need to be the same instance
        with self.assertRaisesRegex(ValueError, "there is already an active mode"):
            with enable_torch_dispatch_mode(LoggingTensorMode()):
                with enable_torch_dispatch_mode(LoggingTensorMode()):
                    pass

    def test_nesting_with_same_enable_torch_dispatch_mode(self) -> None:
        # "nested" enable_torch_dispatch_modes are allowed if they're the same mode (same instance).
        # It's the equivalent of a noop, so it will only write once to the log
        x = torch.tensor([3.])
        mode = LoggingTensorMode()
        with capture_logs(is_mode=True) as logs:
            log_input("x", x)
            with enable_torch_dispatch_mode(mode):
                with enable_torch_dispatch_mode(mode):
                    x + x
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten.add.Tensor($0, $0)''')

    def test_enable_torch_dispatch_mode_ignore_preexisting(self):
        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise AssertionError

        x = torch.tensor([3.])
        with capture_logs(is_mode=True) as logs:
            with enable_torch_dispatch_mode(A()):
                with enable_torch_dispatch_mode(LoggingTensorMode(), ignore_preexisting=True):
                    x + x
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add.Tensor($0, $0)""")

    def test_enable_torch_dispatch_mode_replace(self):
        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise AssertionError

        x = torch.tensor([3.])
        outer_mode = A()
        with capture_logs(is_mode=True) as logs:
            with enable_torch_dispatch_mode(outer_mode):
                with enable_torch_dispatch_mode(LoggingTensorMode(), replace=outer_mode):
                    x + x
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add.Tensor($0, $0)""")

    def test_exception_handling(self):
        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func.__name__ == 'randn.default':
                    raise RuntimeError()
                return cls(torch.zeros(()))

        with enable_torch_dispatch_mode(A):
            try:
                torch.randn(())
            except RuntimeError:
                pass
            self.assertTrue(isinstance(torch.zeros(()), A))

    def test_ctor_no_inner(self):
        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.zeros([])

        with enable_torch_dispatch_mode(A()):
            x = torch.randn((3, 4))

        self.assertEqual(x, torch.zeros([]))

    def test_with_mode(self):
        class ErrorA(RuntimeError):
            pass

        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise ErrorA()

        with self.assertRaises(ErrorA):
            with A():
                torch.empty([])

    def test_with_mode_created_separately(self):
        class ErrorA(RuntimeError):
            pass

        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise ErrorA()

        x = A()
        with self.assertRaises(ErrorA):
            with x:
                torch.empty([])

    def test_with_nested_modes(self):
        class ErrorA(RuntimeError):
            def __init__(self, msg):
                return super().__init__(msg)

        class A(TorchDispatchMode):
            def __init__(self, msg):
                self.msg = msg

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise ErrorA(self.msg)

        with self.assertRaisesRegex(ErrorA, "layer2"):
            with A("layer1"):
                with A("layer2"):
                    torch.empty([])

    def test_make_subclass_with_modes(self):
        class ModeTensor(torch.Tensor):
            def __new__(cls, elem, mode):
                r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
                r.elem = elem
                r.mode = mode
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                modes = tuple(arg.mode for arg in args + tuple(kwargs.values()) if isinstance(arg, ModeTensor))
                assert all_same_mode(modes)
                with Mode():
                    return func(*args, **kwargs)

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                def unwrap(e):
                    if isinstance(e, ModeTensor):
                        return e.elem
                    else:
                        return e

                def wrap(t):
                    if isinstance(t, torch.Tensor):
                        return ModeTensor(t, self)
                    else:
                        return t

                return wrap(func(*tuple(unwrap(a) for a in args), **kwargs))

        class BasicMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **kwargs)

        x = torch.tensor(4.)
        with Mode():
            y = x + x
            z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)
        self.assertIsInstance(torch.add(y, z), ModeTensor)

        with Mode():
            with BasicMode():  # we can't nest two modes that call make_subclass because it only accepts vanilla tensors
                y = x + x
                z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)
        self.assertIsInstance(torch.add(y, z), ModeTensor)

        assert self.assertRaisesRegex(RuntimeError, "subclass Mode but.* associated to a python object of type Mode")

    def test_notimplemented_mode(self):
        sub_count = 0

        class PoliteMode(TorchDispatchMode):
            def __init__(self):
                self.pre_count = 0
                self.post_count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.pre_count += 1
                if any(t is not torch.Tensor for t in types):
                    return NotImplemented
                self.post_count += 1
                return func(*args, **kwargs)

        class SubTensor(torch.Tensor):
            def __new__(cls, elem):
                r = torch.Tensor._make_wrapper_subclass(cls, elem.shape)
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                nonlocal sub_count
                sub_count += 1

                def unwrap(t):
                    if isinstance(t, SubTensor):
                        return t.elem
                    else:
                        return t

                return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

            __torch_function__ = torch._C._disabled_torch_function_impl

        a = SubTensor(torch.randn(2))
        with PoliteMode() as mode:
            a.abs()

        self.assertEqual(mode.pre_count, 2)
        self.assertEqual(mode.post_count, 1)
        self.assertEqual(sub_count, 1)

        # make sure this doesn't error
        with PoliteMode():
            with PoliteMode():
                a.abs()

    def test_disable_mode(self):
        class FailEverythingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise RuntimeError("arf")

        with FailEverythingMode() as m:
            self.assertRaises(RuntimeError, lambda: torch.ones([2, 3]))
            with enable_torch_dispatch_mode(None, replace=m):
                torch.ones([2, 3])

    def test_make_wrapper_subclass_with_modes(self):
        class ModeTensor(torch.Tensor):
            def __new__(cls, elem, mode):
                r = torch.Tensor._make_wrapper_subclass(cls, elem.shape)
                r.elem = elem
                r.mode = mode
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                modes = (arg.mode for arg in args + tuple(kwargs.values()) if isinstance(arg, ModeTensor))
                outermost = find_outermost_mode(modes)
                with outermost.restore():
                    return func(*args, **kwargs)

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                def unwrap(e):
                    if isinstance(e, ModeTensor):
                        return e.elem
                    else:
                        return e

                def wrap(t):
                    if isinstance(t, torch.Tensor):
                        return ModeTensor(t, self)
                    else:
                        return t

                return wrap(func(*tuple(unwrap(a) for a in args), **kwargs))

        x = torch.tensor(4.)
        with Mode():
            y = x + x
            z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)

        with Mode():
            with Mode():
                y = x + x
                z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)

    def test_error_with_same_mode(self):
        # If the pushed mode is the same instance as the current mode, we allow pushing an already active mode.

        class A(TorchDispatchMode):
            pass

        with A() as a:
            with self.assertRaisesRegex(RuntimeError, "already active in the mode stack"):
                with a:
                    pass

    def test_error_mixing_with_enable(self):
        class A(TorchDispatchMode):
            pass

        with A():
            with self.assertRaisesRegex(ValueError,
                                        "Attempted to enable_torch_dispatch_mode, but there is already an active mode"):
                with enable_torch_dispatch_mode(A()):
                    pass

    def test_allow_mixing_enable_with(self):
        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.ops.aten.sub(args[0], args[1])

        x = torch.tensor([3.])
        with capture_logs(is_mode=True) as logs:
            log_input("x", x)
            with enable_torch_dispatch_mode(LoggingTensorMode()):
                with A():
                    x + x
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten.sub.Tensor($0, $0)''')  # sub because A changed it before hitting logging

    def test_error_on_same_mode(self):
        # If the pushed mode isn't the outermost mode, we error
        class A(TorchDispatchMode):
            pass

        with A() as reenabled:
            with A():
                with self.assertRaisesRegex(RuntimeError, "already active in the mode stack"):
                    with reenabled:
                        pass

    def test_error_using_class_method_on_mode(self):
        class A(TorchDispatchMode):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return func(args, kwargs)

        x = torch.tensor(5.)
        with self.assertRaisesRegex(RuntimeError, "should be a normal method not a class method"):
            with A():
                x + x

    def test_all_same_mode(self):
        x = LoggingTensorMode()
        y = LoggingTensorMode()
        self.assertTrue(all_same_mode([x, x, x]))
        self.assertFalse(all_same_mode([x, None]))
        self.assertFalse(all_same_mode([x, y]))

    def test_tolist_numpy_with_torch_dispatch_mode(self) -> None:
        x = LoggingTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(RuntimeError, "is not supported for tensor subclasses."):
            x.tolist()
        with self.assertRaisesRegex(RuntimeError, "is not supported for tensor subclasses."):
            x.numpy()
        with self.assertRaises(AssertionError):
            self.assertEqual(x, None)

    def test_enable_torch_dispatch_mode_subclass_autograd_device_check(self) -> None:
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

                rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
                if func.overloadpacket.__name__ == "add":
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
        s = torch.Storage()
        z = LoggingTensor(torch.empty([]))
        z.set_(s)

    def test_autograd_in_attr(self):
        # We want the wrapped Tensor to require gradients!
        true_t = torch.rand(2, requires_grad=True)
        t = LoggingTensorReentrant(true_t)

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

    def test_dispatch_super_call(self):
        called = []

        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem)

            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called.append(func)
                return super().__torch_dispatch__(func, types, args, kwargs)

        x = torch.randn(2)
        y = torch.randn(2)
        self.assertEqual(SubTensor(x) + SubTensor(y), x + y)
        self.assertEqual(called, [torch.ops.aten.add.Tensor])

    def test_dispatch_super_call_list_arg(self):
        called = []

        class SubTensorWithListArg(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem)

            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called.append(func)
                return super().__torch_dispatch__(func, types, list(args), kwargs)

        x = torch.randn(2)
        self.assertEqual(SubTensorWithListArg(x).neg(), x.neg())
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_dispatch_super_dont_autograd(self):
        called = []

        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called.append(func)
                # This argument still requires grad because it was passed
                # through directly...
                self.assertTrue(args[0].requires_grad)
                r = super().__torch_dispatch__(func, types, args, kwargs)
                # But the output better not require grad, because that means
                # you did autograd again in torch dispatch (oops)
                self.assertFalse(r.requires_grad)
                return r

        x = SubTensor(torch.randn(2, requires_grad=True))
        x.neg()
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_set_data(self):
        called = 0

        class SubTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                nonlocal called
                called += 1
                return super().__torch_dispatch__(func, types, args, kwargs)

        x = SubTensor(torch.empty(2))
        x.data
        self.assertEqual(called, 1)
        x.data = torch.empty(2)
        self.assertEqual(called, 1)
        x.data
        self.assertEqual(called, 2)
        self.assertIs(type(x), SubTensor)
        x.set_(torch.empty(2))
        self.assertEqual(called, 3)
        x.data
        self.assertEqual(called, 4)
        self.assertIs(type(x), SubTensor)

    def test_construct_int_tensor(self):
        class SubTensor(torch.Tensor):
            pass
        # should not fail
        SubTensor(torch.zeros(2, dtype=torch.int))

    def test_multiple_ops_subclass(self):
        # This is a Direct Subclass, don't do that!
        class MySubclass(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                r = torch.Tensor._make_subclass(cls, elem)
                return r

            __torch_function__ = torch._C._disabled_torch_function_impl

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                with no_dispatch():
                    return func(*args, **kwargs)

        x = MySubclass(torch.rand(2, 2, dtype=torch.complex64))
        y = x.conj()
        # Details of the bug that this tests for:
        # Here, y dispatch keys are: {PythonTLSSnapshot, AutogradCPU, Conjugate, Python, CPU}
        # There are a few calls to the dispatcher that are going to happen here:
        #  - call_exp: User calling exp on y
        #    - PythonTLSSnapshot: records the TLS on entry and redispatch
        #    - AutogradCPU: no input requires grad, so does nothing and redispatch
        #    - Conjugate: no special implementation for exp: use the fallback that
        #                 first clone the Tensor (to materialize the conj) then redispatch
        #      - call_clone: conjugate fallback calling clone on y
        #        - PythonTLSSnapshot: records the TLS on entry and redispatch
        #        - (AutogradCPU: skipped as autograd added itself to the exclude set above)
        #        - Conjugate: special implementation for clone: just skip this key
        #        - Python: Reset the TLS based on the snapshot above and call the user implementation (this
        #                  actually calls into the dispatcher again but since we disable both our keys
        #                  before, not detailed here)
        #        - exit Python: restore the TLS and exit
        #        - exit Conjugate: nothing was inplace so just exit
        #        - exit PythonTLSSnapshot: done with this call, reset the saved TLS to empty
        #    - Python: Reset the TLS again based on the snapshot. <- this used to fail
        #    - More steps....
        y.exp()

    @staticmethod
    def subclass_helper(cls, data, use_wrapper_subclass, **kwargs):
        if use_wrapper_subclass:
            kwargs["device"] = data.device
            kwargs["dtype"] = data.dtype
            kwargs["layout"] = data.layout
            kwargs["requires_grad"] = True
            return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)  # type: ignore[attr-defined]
        else:
            return torch.Tensor._make_subclass(cls, data, True, **kwargs)

    def test_is_contiguous_slow_path(self):
        data = torch.randn(3, 3)
        contiguous_data = data.clone()
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))

        for use_wrapper_subclass in [True, False]:
            class ExampleTensor1(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="strides")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="strides")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return contiguous_data.is_contiguous()
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="strides")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return not_contiguous_data.is_contiguous()
                    return NotImplemented

            err_msg = "no implementation found for 'torch.ops.aten.is_contiguous'"
            e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.is_contiguous()
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

            e = ExampleTensor2(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), True)
            e.contiguous()  # this will just return the original TensorImpl since is_contiguous = True

            err_msg = "no implementation found for"
            e = ExampleTensor3(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), False)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

    def test_device_slowpath(self):
        for use_wrapper_subclass in [True]:
            class ExampleTensor1(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_device=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_device=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device('meta')
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_device=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device('meta')
                    return NotImplemented

            err_msg = "no implementation found for 'torch.ops.prim.device'"
            with self.assertRaisesRegex(TypeError, err_msg):
                e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
                e.device()

            ten = torch.rand([1])
            e = ExampleTensor2(torch.randn(3, 3, device='cpu'), use_wrapper_subclass)
            self.assertEqual(e.device.type, 'meta')
            self.assertEqual(ten.type_as(e).device.type, 'meta')

            e = ExampleTensor3(torch.randn(3, 3, device='cpu'), use_wrapper_subclass)
            self.assertEqual(e.device.type, 'meta')
            self.assertEqual(ten.type_as(e).device.type, 'meta')

    def test_dim_slowpath(self):
        data = torch.randn(3, 3)

        for use_wrapper_subclass in [True, False]:
            class DimNotImplementedTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="sizes")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class DimImplementedTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="sizes")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented

            err_msg = "no implementation found for 'torch.ops.aten.dim'"
            e = DimNotImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.dim()

            t = DimImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(t.dim(), 2)

    def test_maybe_tuple_bug(self):
        class T(torch.Tensor):
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                pass
        a = torch.rand(3)

        a[[T(), T()]]

    def test_standard_is_not_subclass(self):
        # https://github.com/pytorch/pytorch/issues/79079
        self.assertFalse(torch._C._dispatch_isTensorSubclassLike(torch.empty(0)))

    def test_strides_slow_path(self):
        for use_wrapper_subclass in [True, False]:
            class StridesNotImplemented(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="strides")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class StridesCustomReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="strides")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func == torch.ops.aten.stride.default:
                        return (4, 2)
                    return NotImplemented

            class StridesDefaultReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="strides")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func == torch.ops.aten.stride.default:
                        return None
                    return NotImplemented

            err_msg = "no implementation found for 'torch.ops.aten.stride'"
            e = StridesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.stride()

            e = StridesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.stride(), (4, 2))

            e = StridesDefaultReturn(torch.randn(6, 2), use_wrapper_subclass)
            self.assertEqual(e.stride(), (2, 1))

    def test_sizes_slow_path(self):
        for use_wrapper_subclass in [True, False]:
            data = torch.randn(6, 2)

            class SizesNotImplemented(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="sizes")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented

            class SizesCustomReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="sizes")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return (5, 3)
                    return NotImplemented

            class SizesDefaultReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="sizes")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return None
                    return NotImplemented

            err_msg = "no implementation found for 'torch.ops.aten.sym_size'"
            e = SizesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                e.size()

            e = SizesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.size(), (5, 3))

            e = SizesDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.size(), (4, 2))

    def test_layout_slow_path(self):
        for use_wrapper_subclass in [True, False]:
            data = torch.randn(6, 2)

            class LayoutNotImplemented(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_layout=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class LayoutCustomReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_layout=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.layout:
                        return torch.sparse_csr
                    return NotImplemented

            class LayoutDefaultReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_layout=True)

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.layout:
                        return data.layout
                    return NotImplemented

            err_msg = "no implementation found for 'torch.ops.prim.layout'"
            e = LayoutNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.layout

            e = LayoutCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.sparse_csr)

            e = LayoutDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.strided)

if __name__ == '__main__':
    run_tests()
