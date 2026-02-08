# Owner(s): ["module: __torch_dispatch__"]
# ruff: noqa: F841

import gc
import pickle
import sys
import tempfile
import unittest
import weakref
from copy import deepcopy

import torch
import torch._dynamo
from torch import SymInt
from torch._C import DispatchKey, DispatchKeySet
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda.jiterator import _create_jit_fn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.library import _scoped_library, fallthrough_kernel, impl, Library
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    first_sample,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.logging_tensor import (
    capture_logs,
    capture_logs_with_logging_tensor_mode,
    log_input,
    LoggingTensor,
    LoggingTensorMode,
    LoggingTensorReentrant,
)
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import all_same_mode, no_dispatch
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    is_in_torch_dispatch_mode,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map, tree_map_only


# used as DataLoader collate_fn below; named here to avoid trying to pickle a lambda
def _identity(x):
    return x


class TestDispatcherPythonBindings(TestCase):
    def test_call_boxed(self) -> None:
        sin = torch._C._dispatch_find_schema_or_throw("aten::sin", "")
        x = torch.randn(3)
        y = torch._C._dispatch_call_boxed(sin, x)
        self.assertEqual(y, x.sin())


class TestPythonRegistration(TestCase):
    test_ns = "_test_python_registration"

    def tearDown(self):
        if hasattr(torch.ops, self.test_ns):
            del torch.ops._test_python_registration

    def test_global_enter(self):
        try:
            v = LoggingTensorMode()
            v_ref = weakref.ref(v)

            v.__enter__()
            # The bug trigger when the C++ stack is the only
            # owner of the mode object.
            del v

            # Does not segfault
            str(torch.rand(2))

        finally:
            v_ref().__exit__(None, None, None)

    def test_fallback(self) -> None:
        test_key = "TESTING_ONLY_GenericMode"
        test_keyset = torch._C.DispatchKeySet(test_key)
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            expected_op = None
            expected_args = None
            expected_kwargs = None
            # Use this out shape to make sure the result from our fallback
            # is what is returned to the user
            out_shape = None

            def my_fallback(op, *args, **kwargs):
                # Disable our handler during checks and generating the output
                with torch._C._ForceDispatchKeyGuard(
                    include_to_set, exclude_to_set | test_keyset
                ):
                    self.assertIs(op, expected_op)
                    self.assertEqual(args, expected_args)
                    self.assertEqual(kwargs, expected_kwargs)
                    # Return something specific
                    return torch.empty(out_shape)

            my_lib.fallback(my_fallback, test_key)

            a, b = torch.rand(2), torch.rand(2)

            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                # Check a factory function
                expected_op = torch.ops.aten.empty.memory_format
                expected_args = ((2, 2),)
                # Extra kwargs to bypass issues with default args in factory functions
                expected_kwargs = {
                    "dtype": torch.float64,
                    "pin_memory": False,
                    "device": torch.device("cpu"),
                }
                out_shape = (3,)
                out = torch.empty(*expected_args, **expected_kwargs)
                self.assertEqual(out.size(), out_shape)

                # Check a regular function
                expected_op = torch.ops.aten.add.Tensor
                expected_args = (a, b)
                expected_kwargs = {}
                out_shape = (4,)
                out = a + b
                self.assertEqual(out.size(), out_shape)

    def test_fallback_keyset(self) -> None:
        test_key_first = "TESTING_ONLY_GenericMode"
        test_key_second = "TESTING_ONLY_GenericWrapper"
        test_keyset = torch._C.DispatchKeySet(test_key_first) | torch._C.DispatchKeySet(
            test_key_second
        )
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            first_called = False
            second_called = False

            def first_fallback(keyset, op, *args, **kwargs):
                nonlocal first_called
                if second_called:
                    # Recursive call
                    first_called = True
                    with torch._C._ForceDispatchKeyGuard(
                        include_to_set, exclude_to_set | test_keyset
                    ):
                        return op(*args, **kwargs)
                else:
                    # Redispatch down
                    keyset = keyset.remove(test_key_first)
                    return op.redispatch(keyset, *args, **kwargs)

            def second_fallback(op, *args, **kwargs):
                nonlocal second_called
                # Set to avoid infinite recursion
                second_called = True
                # New dispatcher call should hit the first callback again
                self.assertFalse(first_called)
                a, b = args
                # Make a subtraction here instead of add !
                c = a - b
                self.assertTrue(first_called)
                return c

            my_lib.fallback(first_fallback, test_key_first, with_keyset=True)
            my_lib.fallback(second_fallback, test_key_second)

            a, b = torch.rand(2), torch.rand(2)
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                c = a + b

            self.assertEqual(c, a - b)
            self.assertTrue(first_called)
            self.assertTrue(second_called)

    def test_fallback_fallthrough(self) -> None:
        test_key_first = "TESTING_ONLY_GenericMode"
        test_key_second = "TESTING_ONLY_GenericWrapper"
        test_keyset = torch._C.DispatchKeySet(test_key_first) | torch._C.DispatchKeySet(
            test_key_second
        )
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            is_called = False

            def my_fallback(op, *args, **kwargs):
                nonlocal is_called
                is_called = True
                with torch._C._ForceDispatchKeyGuard(
                    include_to_set, exclude_to_set | test_keyset
                ):
                    return op(*args, **kwargs)

            my_lib.fallback(torch.library.fallthrough_kernel, test_key_first)
            my_lib.fallback(my_fallback, test_key_second)

            a, b = torch.rand(2), torch.rand(2)
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                c = a + b

            self.assertEqual(c, a + b)
            self.assertTrue(is_called)

    @unittest.skip(
        "Causing flakiness, see https://github.com/pytorch/pytorch/issues/145108"
    )
    def test_fallthrough_for_dense_key_with_meta_in_tls(self) -> None:
        # This tests that if meta is included in TlS dispatch key set,
        # then a meta kernel should be called regardless if a dense
        # backend has a fallthrough kernel

        a = torch.randn((3, 3))
        with _scoped_library("custom", "DEF") as my_lib:
            my_lib.define("sum(Tensor self) -> Tensor")
            meta_is_called = False

            def sum_meta(*args, **kwargs):
                nonlocal meta_is_called
                meta_is_called = True
                return args[0]

            my_lib.impl("sum", fallthrough_kernel, "CPU")
            my_lib.impl("sum", sum_meta, "Meta")

            with torch._C._IncludeDispatchKeyGuard(torch.DispatchKey.Meta):
                torch.ops.custom.sum.default(a)
                self.assertTrue(meta_is_called)

    def test_dispatchkeyset_pickle(self) -> None:
        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        serialized = pickle.dumps(keyset)
        new_keyset = pickle.loads(serialized)
        self.assertEqual(new_keyset, keyset)

    def test_dispatchkeyset_eq(self) -> None:
        a = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        b = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        c = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a != c)

    def test_override_aten_ops_with_multiple_libraries(self) -> None:
        x = torch.tensor([1, 2])
        with _scoped_library("aten", "IMPL") as my_lib2:
            with _scoped_library("aten", "IMPL") as my_lib1:
                # Example 1
                def my_neg(*args, **kwargs):
                    return args[0]._neg_view()

                # Now we are secretly making the operator a view op so autograd needs to know how
                # to handle it
                my_lib1.impl("neg", my_neg, "AutogradCPU")

                self.assertTrue(torch.neg(x).is_neg())

                # RuntimeError: impl("aten::neg", ...):
                # Explicitly provided namespace (aten) in operator name does not match ...
                with self.assertRaisesRegex(
                    RuntimeError, "operator name does not match namespace"
                ):
                    with _scoped_library("foo", "DEF") as my_lib3:
                        my_lib3.define("neg(Tensor self) -> Tensor")
                        my_lib3.impl(torch.ops.aten.neg.default, my_neg, "AutogradCPU")

                # Example 2
                def my_mul(*args, **kwargs):
                    return torch.zeros_like(args[0])

                # torch.ops.aten.mul.Tensor
                my_lib2.impl("aten::mul.Tensor", my_mul, "ZeroTensor")

                y = torch._efficientzerotensor(2)
                self.assertFalse(torch.mul(x, y)._is_zerotensor())

                # Assert that a user can't override the behavior of a (ns, op, dispatch_key)
                # combination if someone overridden the behavior for the same before them
                with self.assertRaisesRegex(
                    RuntimeError, "already a kernel registered from python"
                ):
                    my_lib2.impl(torch.ops.aten.mul.Tensor, my_mul, "ZeroTensor")

            # Validate that lib2 is not affected by removing lib1
            self.assertFalse(torch.mul(x, y)._is_zerotensor())

        # Validate that the old behavior is restored for neg and mul
        self.assertFalse(torch.neg(x).is_neg())
        self.assertTrue(torch.mul(x, y)._is_zerotensor())

    def test_error_if_fn_not_callable(self):
        with self.assertRaisesRegex(
            TypeError, "Input function is required to be a callable"
        ):
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl(torch.ops.aten.neg.default, [], "AutogradCPU")

    def test_finalizer(self):
        impls_refcnt = sys.getrefcount(torch.library._impls)
        lib = Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        lib.define("foo123(Tensor x) -> Tensor")

        # 1 for `lib`, 1 for sys.getrefcount' for previous python version (<=3.12)
        # In Python 3.13+, sys.getrefcount() was optimized to not create
        # a temporary reference, so expected counts are 1 less than before
        expected_refcount = 1 if sys.version_info >= (3, 14) else 2
        self.assertEqual(sys.getrefcount(lib), expected_refcount)

        # We gained an additional reference that gets cleared when the finalizer runs
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt + 1)
        # 1 for `lib`
        # 1 for the finalizer
        # 1 for sys.getrefcount
        self.assertEqual(sys.getrefcount(lib._op_impls), 3)

        def foo123(x):
            pass

        lib.impl(f"{self.test_ns}::foo123", foo123, "CPU")
        key = f"{self.test_ns}/foo123/CPU"
        self.assertTrue(key in torch.library._impls)

        saved_op_impls = lib._op_impls

        # del will definitely work if the following passes
        self.assertEqual(sys.getrefcount(lib), expected_refcount)
        del lib

        # 1 for saved_op_impls
        # 1 for sys.getrefcount
        # This function should be the last user of lib._op_impls:
        # - lib should not have a reference anymore (it was del'ed)
        # - lib's finalizer should not have a reference anymore
        self.assertEqual(sys.getrefcount(saved_op_impls), expected_refcount)

        self.assertTrue(key not in torch.library._impls)

        # lib's finalizer should not have a reference anymore
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt)

    def test_override_cpu_sum(self) -> None:
        # Example 1
        run = [False]

        def my_sum(*args, **kwargs):
            run[0] = True
            return args[0].clone()

        with _scoped_library("aten", "IMPL") as my_lib1:
            my_lib1.impl("aten::sum", my_sum, "CPU")
            x = torch.tensor([1, 2])
            self.assertEqual(torch.sum(x), x)
            self.assertTrue(run[0])
        # Validate that the old behavior is restored for sum
        self.assertEqual(torch.sum(x), torch.tensor(3))

    def test_override_cuda_with_jiterator(self) -> None:
        def override_where_cuda() -> None:
            # Example 1: Invert the behavior of where's condition input
            not_where_code_string = """
            template <typename T> T inverted_where(bool cond, T a, T b){
                return !cond ? a : b;
            }
            """
            jitted_where = _create_jit_fn(not_where_code_string)

            CALLED = [False]

            def inverted_where(*args, **kwargs):
                CALLED[0] = True
                return jitted_where(*args, **kwargs)

            # overriding where's cuda kernel with Jiterator generated kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::where.self", inverted_where, "CUDA")

                device = "cuda"
                cond = torch.tensor(
                    [True, True, False], device=device, dtype=torch.bool
                )
                x = torch.tensor([1, 2, 3], device=device)
                y = torch.tensor([-1, -2, -3], device=device)

                self.assertEqual(torch.where(cond, x, y), torch.tensor([-1, -2, 3]))
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(torch.where(cond, x, y), torch.tensor([1, 2, -3]))

        def override_gelu_cuda() -> None:
            # Example 2: Use relu to approximate gelu for faster compute
            fastest_gelu_code_string = """
            template <typename T> T fast_gelu(T a){
                return a > 0 ? a : 0;
            }
            """
            jitted_gelu = _create_jit_fn(fastest_gelu_code_string)

            CALLED = [False]

            def fast_gelu(*args, **kwargs):
                CALLED[0] = True
                return jitted_gelu(*args, **kwargs)

            # overriding gelu's cuda kernel with Jiterator generated relu kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::gelu", fast_gelu, "CUDA")

                x = torch.rand([3, 3], device="cuda", dtype=torch.float)
                self.assertEqual(
                    torch.nn.functional.gelu(x), torch.nn.functional.relu(x)
                )
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertNotEqual(
                torch.nn.functional.gelu(x), torch.nn.functional.relu(x)
            )

        def override_exp_cuda() -> None:
            # Example 3: Preventing exp from exploding for float16
            clipped_exp_code_string = """
            template <typename T> T clipped_exp(T a){
                return a > T(10.0) ? T(22026.4657948) : exp(a);
            }
            """
            jitted_exp = _create_jit_fn(clipped_exp_code_string)

            CALLED = [False]

            def clipped_exp(*args, **kwargs):
                CALLED[0] = True
                return jitted_exp(*args, **kwargs)

            # overriding exp's cuda kernel with clipped_exp kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::exp", clipped_exp, "CUDA")

                x = torch.tensor([0.0, 100.0], device="cuda", dtype=torch.float16)
                self.assertEqual(
                    torch.exp(x),
                    torch.tensor([1.0, 22026.4657948], dtype=torch.float16),
                )
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(
                torch.exp(x), torch.tensor([1.0, torch.inf], dtype=torch.float16)
            )

        def override_add_cuda() -> None:
            # Example 4: simulate a hardware bug, where the adder is always off by 1
            buggy_add_code_string = """
            template <typename T> T buggy_add(T a, T b){
                return a + b + T(1);
            }
            """
            jitted_add = _create_jit_fn(buggy_add_code_string)

            CALLED = [False]

            def buggy_add(*args, **kwargs):
                CALLED[0] = True
                return jitted_add(*args, **kwargs)

            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::add.Tensor", buggy_add, "CUDA")

                x_cpu = torch.rand([3, 3], device="cpu")
                y_cpu = torch.rand([3], device="cpu")

                x_cuda = x_cpu.cuda()
                y_cuda = y_cpu.cuda()

                self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu + 1)
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu)

        if torch.cuda.is_available() and not TEST_WITH_ROCM:
            override_where_cuda()
            override_gelu_cuda()
            override_exp_cuda()
            override_add_cuda()

    def test_extend_library_with_dispatch_key_arg(self):
        def my_sum(*args, **kwargs):
            return args[0].clone()

        with _scoped_library("aten", "IMPL", dispatch_key="CPU") as my_lib1:
            # RuntimeError: Explicitly provided dispatch key (Conjugate) is
            # inconsistent with the dispatch key of the enclosing TORCH_LIBRARY_IMPL block
            with self.assertRaisesRegex(
                RuntimeError, "inconsistent with the dispatch key"
            ):
                my_lib1.impl("sum", my_sum, "Conjugate")
            my_lib1.impl("aten::sum", my_sum)
            x = torch.tensor([1, 2])
            self.assertEqual(torch.sum(x), x)

    def test_create_new_library(self) -> None:
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            my_lib1.define("sum(Tensor self) -> Tensor")

            # Example 1
            @torch.library.impl(my_lib1, "sum", "CPU")
            def my_sum(*args, **kwargs):
                return args[0].clone()

            x = torch.tensor([1, 2])
            op = getattr(torch.ops, self.test_ns).sum
            self.assertEqual(op(x), x)

            with _scoped_library(self.test_ns, "IMPL") as my_lib2:
                # Example 2
                @torch.library.impl(my_lib2, op.default, "ZeroTensor")
                def my_sum_zt(*args, **kwargs):
                    if args[0]._is_zerotensor():
                        return torch._efficientzerotensor(args[0].shape)
                    else:
                        return args[0].clone()

                y = torch._efficientzerotensor(3)
                self.assertTrue(op(y)._is_zerotensor())
                self.assertEqual(op(x), x)

    def test_create_new_library_fragment_no_existing(self):
        with _scoped_library(self.test_ns, "FRAGMENT") as my_lib:
            my_lib.define("sum2(Tensor self) -> Tensor")

            @torch.library.impl(my_lib, "sum2", "CPU")
            def my_sum(*args, **kwargs):
                return args[0]

            x = torch.tensor([1, 2])
            self.assertEqual(getattr(torch.ops, self.test_ns).sum2(x), x)

    def test_create_new_library_fragment_with_existing(self):
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            # Create a fragment
            with _scoped_library(self.test_ns, "FRAGMENT") as my_lib2:
                my_lib2.define("sum4(Tensor self) -> Tensor")

                @torch.library.impl(my_lib2, "sum4", "CPU")
                def my_sum4(*args, **kwargs):
                    return args[0]

                x = torch.tensor([1, 2])
                self.assertEqual(getattr(torch.ops, self.test_ns).sum4(x), x)

                # Create another fragment
                with _scoped_library(self.test_ns, "FRAGMENT") as my_lib3:
                    my_lib3.define("sum3(Tensor self) -> Tensor")

                    @torch.library.impl(my_lib3, "sum3", "CPU")
                    def my_sum3(*args, **kwargs):
                        return args[0]

                    x = torch.tensor([1, 2])
                    self.assertEqual(getattr(torch.ops, self.test_ns).sum3(x), x)

    @unittest.skipIf(IS_WINDOWS, "Skipped under Windows")
    def test_alias_analysis(self):
        def test_helper(alias_analysis=""):
            my_lib1 = Library(self.test_ns, "DEF")  # noqa: TOR901

            called = [0]

            @torch.library.define(
                my_lib1, "_op() -> None", alias_analysis=alias_analysis
            )
            def _op(*args, **kwargs):
                called[0] += 1

            @torch.jit.script
            def _test():
                torch.ops._test_python_registration._op()

            assert "_test_python_registration::_op" in str(_test.graph)

        with self.assertRaises(AssertionError):
            test_helper("")  # alias_analysis="FROM_SCHEMA"

        # Run gc to make sure the previous Library is removed.  This is needed in dynamo-wrapped 3.14t
        gc.collect()
        test_helper("CONSERVATIVE")

    def test_error_for_unsupported_ns_or_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported kind"):
            my_lib1 = Library("myns", "BLA")  # noqa: TOR901

        for kind in ("DEF", "FRAGMENT"):
            with self.assertRaisesRegex(ValueError, "reserved namespace"):
                my_lib1 = Library("prim", kind)  # noqa: TOR901

    def test_dispatcher_error_filenames(self) -> None:
        # Test that dispatcher errors report correct Python filenames and line numbers
        # when defining duplicate libraries (which triggers the filename tracking)
        import linecache
        import re

        # Create first library
        # NOTE: Using Library directly instead of _scoped_library because this test
        # specifically verifies filename tracking in error messages, and _scoped_library
        # would report library.py locations instead of the actual test file locations
        lib1 = Library(self.test_ns, "DEF")  # FIRST_LIB_MARKER  # noqa: TOR901
        try:
            lib1.define("duplicate_op(Tensor x) -> Tensor")

            # Try to create another library with same namespace - this should trigger error
            with self.assertRaises(RuntimeError) as cm:
                lib2 = Library(self.test_ns, "DEF")  # SECOND_LIB_MARKER  # noqa: TOR901
        finally:
            lib1._destroy()

        error_msg = str(cm.exception)

        # The error should NOT contain /dev/null (the old placeholder)
        self.assertNotIn("/dev/null", error_msg)
        # The error should contain the test file name for both registrations
        self.assertIn("test_python_dispatch.py", error_msg)
        # Extract line numbers from the error message and verify they point to the right lines
        line_matches = re.findall(r"test_python_dispatch\.py:(\d+)", error_msg)
        self.assertEqual(
            len(line_matches), 2, "Should have exactly 2 line number references"
        )

        # Get the actual source lines and verify they contain our markers
        first_line_num, second_line_num = sorted([int(x) for x in line_matches])
        first_line = linecache.getline(__file__, first_line_num).strip()
        second_line = linecache.getline(__file__, second_line_num).strip()

        # Verify the lines contain our expected markers
        self.assertIn("FIRST_LIB_MARKER", first_line)
        self.assertIn("SECOND_LIB_MARKER", second_line)

    def test_returning_symint(self) -> None:
        shape_env = ShapeEnv()
        fake_tensor_mode = FakeTensorMode(shape_env=shape_env)

        ft = fake_tensor_mode.from_tensor(torch.rand(2, 3))

        s0, s1 = ft.shape

        with _scoped_library(self.test_ns, "DEF") as tlib:
            tlib.define("sqsum(SymInt a, SymInt b) -> SymInt")

            @impl(tlib, "sqsum", "CompositeExplicitAutograd")
            def sqsum(a: SymInt, b: SymInt):
                return a * a + b * b

            out = getattr(torch.ops, self.test_ns).sqsum.default(s0, s1)
            out_val = shape_env.evaluate_expr(out.node.expr)
        self.assertEqual(out_val, 13)

    def test_register_fallthrough(self):
        with _scoped_library("aten", "IMPL") as my_lib:
            my_lib.impl("mm", fallthrough_kernel, "AutocastCPU")

            a = torch.randn(2, 3, device="cpu", dtype=torch.float32)
            b = torch.randn(3, 2, device="cpu", dtype=torch.float32)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                # dtype for mm should be float32 since we registered a fallthrough
                self.assertEqual(torch.mm(a, b).dtype, torch.float32)
                # ops that don't have a fallthrough registered should not be affected
                self.assertEqual(torch.matmul(a, b).dtype, torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # default behavior should have been restored
            self.assertEqual(torch.mm(a, b).dtype, torch.bfloat16)


class TestPythonDispatch(TestCase):
    def test_basic(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input("x", x)
            y = x * x
            saved_x = y.grad_fn._saved_self
            grad_y = LoggingTensor(torch.tensor([1.0]))
            log_input("grad_y", grad_y)
            (g,) = torch.autograd.grad((y,), (x,), (grad_y,))

        self.assertEqual(g.elem, torch.tensor([6.0]))
        with torch.no_grad():
            self.assertEqual(saved_x, x)
            self.assertEqual(saved_x._version, x._version)
            x.add_(2)
            self.assertEqual(saved_x, x)
            # TODO: figure out why broken
            # self.assertEqual(saved_x._version, x._version)
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten.mul.Tensor($0, $0)
$2: f32[1] = input('grad_y')
$3: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$4: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$5: f32[1] = torch._ops.aten.add.Tensor($4, $3)""",
        )

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
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = input('y')
$2: f32[1] = torch._ops.aten.abs.out($0, out=$1)""",
        )

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
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1, 1] = input('y')
$2: f32[1] = input('z')
$3: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$4: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$5: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2)
$6: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, alpha=2)
$7: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2, alpha=2)""",
        )

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
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten._foobar.default($0)
$2: f32[1] = torch._ops.aten._foobar.default($0, False)
$3: f32[1] = torch._ops.aten._foobar.default($0, arg3=False)
$4: f32[1] = torch._ops.aten._foobar.default($0, False, arg3=False)""",
        )

    def test_produce_real_type(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(2, 2))
            log_input("x", x)
            x.to(dtype=torch.double)  # non-optional dtype
            torch.cumprod(x, 0, dtype=torch.double)  # optional dtype
            x[:, 1].contiguous(
                memory_format=torch.contiguous_format
            )  # optional memory format
            # There doesn't appear to be any layout signatures which are
            # triggerable using tensor subclasses (need to use a mode)

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[2, 2] = input('x')
$1: f64[2, 2] = torch._ops.aten._to_copy.default($0, dtype=torch.float64)
$2: f64[2, 2] = torch._ops.aten.cumprod.default($0, 0, dtype=torch.float64)
$3: f32[2] = torch._ops.aten.select.int($0, 1, 1)
$4: f32[2] = torch._ops.aten.clone.default($3, memory_format=torch.contiguous_format)""",
        )

    def test_optional_tensor_list(self) -> None:
        def weird(xs):
            print("woof")
            return torch.empty(())

        with _scoped_library("my_lib", "DEF") as my_lib:
            my_lib.define("weird(Tensor?[] self) -> Tensor")
            my_lib.impl("weird", weird, "CPU")
            with capture_logs() as logs:
                x = LoggingTensor(torch.ones(2, 2))
                log_input("x", x)
                torch.ops.my_lib.weird.default([None, x])

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[2, 2] = input('x')
$1: f32[] = torch._ops.my_lib.weird.default(['None', '$0'])""",
        )

    def test_list_ret(self) -> None:
        # test all sequence types are permissible returns
        for list_type in (list, tuple):

            class A(torch.Tensor):
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
                torch.split(torch.tensor([0, 1]), 2),
            )

    def test_invalid_ret(self) -> None:
        # test invalid return gets reasonable error message
        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return "arf"

        # Wobbles depending on NDEBUG mode of pybind11
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).neg(),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).detach(),
        )

    def test_detach_appears_once_when_called_once(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input("x", x)
            x.detach()
        # FIXME: We actually want this to emit a single detach. However,
        # it currently emits two, for reasons unclear to us. Leaving
        # this test here to make sure we don't regress even further (it
        # would be bad if calling .detach() once emits 3+ detaches).
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten.detach.default($0)""",
        )

    def test_storage(self) -> None:
        # For now, just make sure it doesn't crash.  Ideally, we should
        # return some virtual storage that is safe to work with
        x = LoggingTensor(torch.ones(1))
        storage = x.untyped_storage()
        self.assertRaises(RuntimeError, lambda: storage.data_ptr())

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

        self.assertRaises(
            ErrorA, lambda: torch.add(A(torch.empty(1)), A(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(A(torch.empty(1)), B(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(B(torch.empty(1)), A(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(B(torch.empty(1)), B(torch.empty(1)))
        )

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
                y = x**2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                assert isinstance(grad_output, LoggingTensor)
                (x,) = ctx.saved_tensors
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

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = input('x.grad')
$2: f32[1] = torch._ops.aten.pow.Tensor_Scalar($0, 2)
$3: f32[1] = input('grad_output')
$4: f32[1] = torch._ops.aten.mul.Tensor($3, 2)
$5: f32[1] = torch._ops.aten.mul.Tensor($4, $0)
$6: f32[1] = torch._ops.aten.add_.Tensor($1, $5)""",
        )

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
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return MyTensor(3)

        self.assertEqual(type(MyTensor(2).new_ones(3)), MyTensor)

    def test_like(self) -> None:
        class MyTensor(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return MyTensor(3)

        for f in ["empty", "ones", "rand", "randn", "zeros"]:
            f_name = f + "_like"
            self.assertEqual(type(getattr(torch, f_name)(MyTensor(2))), MyTensor)

        self.assertEqual(type(torch.full_like(MyTensor(2), 1.0)), MyTensor)
        self.assertEqual(type(torch.randint_like(MyTensor(2), high=3)), MyTensor)

    def test_make_fx_with_subclass(self) -> None:
        def f(x, y):
            # Returns (TwoTensor, Tensor)
            return x * y, y + y

        x_a = torch.zeros(4)
        x_b = torch.zeros(4)
        y = torch.ones(4)

        # make_fx() is not responsible for unwrapping tensor subclass inputs,
        # so we do it manually here.
        # Why? In general, make_fx(f)(*args) promises that the graph returned has the same calling
        # convention as f(*args). Unwrapping tensor subclass inputs can potentially change
        # the number of input args to the graph, breaking that assumption
        def f_to_trace(x_a, x_b, y):
            x = TwoTensor(x_a, x_b)
            out1, out2 = f(x, y)
            out1_unwrapped_attrs, _ = out1.__tensor_flatten__()
            return (*[getattr(out1, attr) for attr in out1_unwrapped_attrs], out2)

        fx_g = make_fx(f_to_trace, tracing_mode="fake")(x_a, x_b, y)
        self.assertExpectedInline(
            fx_g.code,
            """\



def forward(self, x_a_1, x_b_1, y_1):
    mul = torch.ops.aten.mul.Tensor(x_a_1, y_1);  x_a_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(x_b_1, y_1);  x_b_1 = None
    add = torch.ops.aten.add.Tensor(y_1, y_1);  y_1 = None
    return (mul, mul_1, add)
    """,
        )

    # See https://github.com/pytorch/pytorch/issues/117794
    def test_return_and_correct_aliasing_gives_correct_stride(self):
        t = TwoTensor(torch.randn(2, 2), torch.randn(2, 2))
        x = torch.randn(2, 2)
        # slicing should result in the same stride for TwoTensor as a dense tensor would give
        self.assertEqual(t[:, 0].stride(), x[:, 0].stride())

    def test_make_wrapper_subclass_propagates_metadata(self) -> None:
        class WrapperTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
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
            # purposefully use int64 to test non-default dtype
            x = LoggingTensor(torch.randperm(3))
            torch.save(x, f)
            f.seek(0)
            with torch.serialization.safe_globals([LoggingTensor]):
                x_loaded = torch.load(f)
            self.assertTrue(type(x_loaded) is type(x))
            self.assertEqual(x, x_loaded)
            self.assertEqual(x.elem, x_loaded.elem)
            self.assertFalse(x is x_loaded)

    def test_deepcopy_wrapper_subclass(self) -> None:
        # purposefully use int64 to test non-default dtype
        x = LoggingTensor(torch.randperm(3))
        x_copy = deepcopy(x)
        self.assertTrue(type(x_copy) is type(x))
        self.assertEqual(x, x_copy)
        self.assertEqual(x.elem, x_copy.elem)
        self.assertFalse(x is x_copy)

    def test_deepcopy_wrapper_subclass_with_clone_returning_different_type(
        self,
    ) -> None:
        class MyWrapperTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
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
        with self.assertRaisesRegex(
            RuntimeError,
            "for which cloning returns another instance of the same subclass",
        ):
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
            with self.assertRaisesRegex(
                RuntimeError,
                "for which that function returns another instance of the same subclass",
            ):
                x_copy = deepcopy(x)

        # Ensure a correctly implemented new_empty() causes deepcopy() to work.
        class SubTensorSuccess(torch.Tensor):
            def new_empty(self, shape):
                return type(self)(shape)

        x = SubTensorSuccess(3)
        x_copy = deepcopy(x)
        self.assertIs(type(x_copy), type(x))

    def test_wrapper_subclass_extra_dispatch_keys(self) -> None:
        class ExtraKeysTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # NB: only the non-kwarg overload of _make_wrapper_subclass supports
                #     extra dispatch keys. We probably want to unify the two APIs
                #     in the future.
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    elem.stride(),
                    elem.storage_offset(),
                    torch.contiguous_format,
                    elem.dtype,
                    elem.layout,
                    elem.device,
                    False,
                    False,
                    None,
                    False,
                    False,
                    DispatchKeySet(DispatchKey.NestedTensor),
                )
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                pass

        x = ExtraKeysTensor(torch.randn(3))
        self.assertTrue(torch._C._dispatch_keys(x).has(DispatchKey.NestedTensor))
        self.assertFalse(
            torch._C._dispatch_keys(x).has(DispatchKey.AutogradNestedTensor)
        )

    def test_wrapper_subclass_multiprocessing_preserves_dtype(self):
        # a and b have dtype of int64, which is purposefully different from the default
        # assumed by _make_wrapper_subclass().
        a = torch.randperm(5)
        b = torch.randperm(5)
        data = TwoTensor(a, b)
        expected_dtype = data.dtype

        loader = torch.utils.data.DataLoader(
            [data, data],
            batch_size=2,
            num_workers=2,
            collate_fn=_identity,
        )
        for batch in loader:
            self.assertEqual(batch[0].dtype, expected_dtype)

    def test_index_put_where_only_index_is_subclass(self) -> None:
        called_funcs = []

        class MyTensor(torch.Tensor):
            elem: torch.Tensor
            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
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

    def test_torch_dispatch_mode_basic(self) -> None:
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                torch.empty([])
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)""",
        )

    def test_torch_dispatch_mode_unrelated_tensors(self) -> None:
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                x + y
        self.assertExpectedInline(
            "\n".join(logs), """$2: f32[] = torch._ops.aten.add.Tensor($0, $1)"""
        )

    def test_nested_push_logging_tensor_mode(self):
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                with LoggingTensorMode():
                    torch.empty([])
                    x + y

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

    def test_capture_logs_with_torch_dispatch_mode(self):
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs_with_logging_tensor_mode() as logs:
            torch.empty([])
            x + y
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

        x = torch.randn([])
        y = torch.randn([])

        with capture_logs_with_logging_tensor_mode() as logs1:
            with capture_logs_with_logging_tensor_mode() as logs2:
                torch.empty([])
                x + y

        self.assertExpectedInline(
            "\n".join(logs2),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

        self.assertEqual(logs1, logs2)

    def test_torch_dispatch_mode_subclass_priority(self) -> None:
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
                with AMode():
                    raise ErrorA

        class B(A):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                with BMode():
                    func(*args, **kwargs)

        class AMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise ErrorA

        class BMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
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
            with AMode():
                b + b
        with self.assertRaises(ErrorB):
            with BMode():
                a + a
        with self.assertRaises(ErrorB):
            with BMode():
                a + b

    def test_mode_with_make_subclass(self):
        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class BasicMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **kwargs)

        x = torch.randn(3)
        with BasicMode():
            y = SubTensor(x)
        self.assertIsInstance(y, SubTensor)

    def test_torch_dispatch_mode_respects_no_dispatch(self) -> None:
        with capture_logs(is_mode=True) as logs1:
            with LoggingTensorMode():
                torch.ones([2, 3])
                with no_dispatch():
                    torch.ones([2, 3])
        with capture_logs(is_mode=True) as logs2:
            with LoggingTensorMode():
                torch.ones([2, 3])
        self.assertEqual(logs1, logs2)

    def test_shallow_copy_and_detach(self) -> None:
        seen = set()
        test_case = self

        class TestMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                tree_map_only(
                    torch.Tensor, lambda t: test_case.assertIn(t, seen), (args, kwargs)
                )
                if kwargs is None:
                    kwargs = {}
                r = func(*args, **kwargs)
                tree_map_only(torch.Tensor, lambda t: seen.add(t), r)
                return r

        with TestMode():
            x = torch.randn(3, requires_grad=True)
            loss = (x * x).sum()
            loss.backward()

    def test_exception_handling(self):
        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class AMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func.__name__ == "randn.default":
                    raise RuntimeError
                return A(torch.zeros(()))

        with AMode():
            try:
                torch.randn(())
            except RuntimeError:
                pass
            self.assertTrue(isinstance(torch.zeros(()), A))

    def test_with_mode_created_separately(self):
        class ErrorA(RuntimeError):
            pass

        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                raise ErrorA

        x = A()
        with self.assertRaises(ErrorA):
            with x:
                torch.empty([])

    def test_with_nested_modes(self):
        class ErrorA(RuntimeError):
            def __init__(self, msg):
                super().__init__(msg)

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
                raise NotImplementedError("Shouldn't be here")

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

        x = torch.tensor(4.0)
        with Mode():
            y = x + x
            z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)

        with Mode():
            with BasicMode():  # we can't nest two modes that call make_subclass because it only accepts vanilla tensors
                y = x + x
                z = y + y
        self.assertIsInstance(y, ModeTensor)
        self.assertIsInstance(z, ModeTensor)

        assert self.assertRaisesRegex(
            RuntimeError,
            "subclass Mode but.* associated to a python object of type Mode",
        )

    def test_notimplemented_mode(self):
        sub_count = 0

        class PoliteMode(TorchDispatchMode):
            def __init__(self) -> None:
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

    def test_nesting_same_mode(self):
        # If the pushed mode is the same instance as the current mode, we allow pushing an already active mode.

        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode() as reenabled:
                with reenabled:
                    torch.empty([])
            self.assertExpectedInline(
                "\n".join(logs),
                """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)""",
            )

    def test_error_using_class_method_on_mode(self):
        class A(TorchDispatchMode):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return func(args, kwargs)

        x = torch.tensor(5.0)
        with self.assertRaisesRegex(
            RuntimeError, "classmethod is not supported, please make it a plain method"
        ):
            with A():
                x + x

    def test_get_cur_mode(self):
        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                pass

        self.assertEqual(_get_current_dispatch_mode(), None)

        with A() as mode1:
            self.assertEqual(_get_current_dispatch_mode(), mode1)

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_dispatch_mode(), mode2)

    def test_get_mode_stack(self):
        class A(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                pass

        self.assertEqual(_get_current_dispatch_mode_stack(), [])

        with A() as mode1:
            self.assertEqual(_get_current_dispatch_mode_stack(), [mode1])

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_dispatch_mode_stack(), [mode1, mode2])

    def test_all_same_mode(self):
        x = LoggingTensorMode()
        y = LoggingTensorMode()
        self.assertTrue(all_same_mode([x, x, x]))
        self.assertFalse(all_same_mode([x, None]))
        self.assertFalse(all_same_mode([x, y]))

    def test_mode_detection(self):
        class InfraMode(TorchDispatchMode):
            @classmethod
            def is_infra_mode(cls):
                return True

        class NonInfraMode(TorchDispatchMode):
            pass

        with InfraMode():
            self.assertTrue(is_in_torch_dispatch_mode())
            self.assertFalse(is_in_torch_dispatch_mode(include_infra_modes=False))
            with NonInfraMode():
                self.assertTrue(is_in_torch_dispatch_mode())
                self.assertTrue(is_in_torch_dispatch_mode(include_infra_modes=False))
                with InfraMode():
                    self.assertTrue(is_in_torch_dispatch_mode())
                    self.assertTrue(
                        is_in_torch_dispatch_mode(include_infra_modes=False)
                    )

                self.assertTrue(is_in_torch_dispatch_mode())
                self.assertTrue(is_in_torch_dispatch_mode(include_infra_modes=False))
            self.assertTrue(is_in_torch_dispatch_mode())
            self.assertFalse(is_in_torch_dispatch_mode(include_infra_modes=False))

        self.assertFalse(is_in_torch_dispatch_mode())
        self.assertFalse(is_in_torch_dispatch_mode(include_infra_modes=False))

    def test_tolist_numpy_with_torch_dispatch_mode(self) -> None:
        x = LoggingTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(
            RuntimeError, "is not supported for tensor subclasses."
        ):
            x.tolist()
        with self.assertRaisesRegex(
            RuntimeError, "is not supported for tensor subclasses."
        ):
            x.numpy()
        with self.assertRaises(AssertionError):
            self.assertEqual(x, None)

    # See https://github.com/pytorch/pytorch/issues/136064
    def test_view_returns_alias_under_torch_dispatch(self):
        class MyMode(TorchDispatchMode):
            def __init__(self, testcase):
                self.testcase = testcase

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **kwargs)
                if func == torch.ops.aten.view.dtype:
                    # view should return a fresh TensorImpl
                    self.testcase.assertTrue(out is not args[0])
                return out

        with MyMode(self):
            x = torch.ones(4, dtype=torch.float32)
            out = x.view(torch.float32)

    def test_record_stream(self) -> None:
        class TestMode(TorchDispatchMode):
            def __init__(self, testcase):
                self.testcase = testcase

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.testcase.assertEqual(func.name(), "aten::record_stream")
                self.testcase.assertIsInstance(args[0], torch.Tensor)
                self.testcase.assertIsInstance(args[1], torch.Stream)
                self.testcase.assertEqual(args[1].stream_id, 1)
                self.testcase.assertEqual(args[1].device_index, 2)
                self.testcase.assertEqual(args[1].device_type, 3)

        t = torch.tensor(5.0)
        s = torch.Stream(stream_id=1, device_index=2, device_type=3)
        with TestMode(self):
            t.record_stream(s)

    def test_return_stream(self) -> None:
        with _scoped_library("test_return_stream", "DEF") as l_def:
            l_def.define("return_stream(Tensor self) -> Stream")
            with _scoped_library("test_return_stream", "IMPL", "CPU") as l_impl:
                l_impl.impl(
                    "return_stream",
                    lambda _: torch.Stream(stream_id=0, device_index=1, device_type=2),
                )

                class TestMode(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        return torch.Stream(stream_id=1, device_index=2, device_type=3)

                t = torch.tensor(5.0)
                s = torch.ops.test_return_stream.return_stream(t)
                self.assertIsInstance(s, torch.Stream)
                self.assertEqual(s.stream_id, 0)
                self.assertEqual(s.device_index, 1)
                self.assertEqual(s.device_type, 2)

                with TestMode():
                    s = torch.ops.test_return_stream.return_stream(t)
                self.assertIsInstance(s, torch.Stream)
                self.assertEqual(s.stream_id, 1)
                self.assertEqual(s.device_index, 2)
                self.assertEqual(s.device_type, 3)

    def test_none_wrapping(self):
        # A Tensor subclass that returns None when doing add
        # See LoggingTensor above for more details on the subclass
        class SubclassWithNone(torch.Tensor):
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                def unwrap(e):
                    return e.elem if isinstance(e, SubclassWithNone) else e

                def wrap(e):
                    return SubclassWithNone(e) if isinstance(e, torch.Tensor) else e

                rs = tree_map(
                    wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                )
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
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return contiguous_data.is_contiguous()
                    if func.overloadpacket == torch.ops.aten.sym_is_contiguous:
                        return torch.ops.aten.sym_is_contiguous(contiguous_data)
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return not_contiguous_data.is_contiguous()
                    if func.overloadpacket == torch.ops.aten.sym_is_contiguous:
                        return torch.ops.aten.sym_is_contiguous(not_contiguous_data)
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.is_contiguous'"
            e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.is_contiguous()
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

            e = ExampleTensor2(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), True)
            e.contiguous()  # this will just return the original TensorImpl since is_contiguous = True

            err_msg = "Multiple dispatch failed for"
            e = ExampleTensor3(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), False)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

    def test_fancy_strides(self):
        calls = []

        class ExampleTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return TestPythonDispatch.subclass_helper(
                    cls, data, False, dispatch_sizes_strides_policy="strides"
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if func in [
                    torch.ops.aten.sym_is_contiguous.default,
                    torch.ops.aten.is_contiguous.default,
                    torch.ops.aten.is_contiguous.memory_format,
                    torch.ops.aten.is_strides_like_format.default,
                    torch.ops.aten.is_non_overlapping_and_dense.default,
                    torch.ops.aten.stride.default,
                ]:
                    calls.append((func, list(args)[1:]))
                    return None
                with no_dispatch():
                    return func(*args, **kwargs)

        e = ExampleTensor(torch.randn(2, 2))
        self.assertFalse(e.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(
            calls, [(torch.ops.aten.is_contiguous.memory_format, [torch.channels_last])]
        )
        calls.clear()
        self.assertFalse(
            torch.ops.aten.is_strides_like_format.default(e, torch.channels_last)
        )
        self.assertEqual(
            calls,
            [(torch.ops.aten.is_strides_like_format.default, [torch.channels_last])],
        )
        calls.clear()
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(e))
        self.assertEqual(
            calls, [(torch.ops.aten.is_non_overlapping_and_dense.default, [])]
        )

    def test_device_slowpath(self):
        for use_wrapper_subclass in [True]:

            class ExampleTensor1(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class ExampleTensor2(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device("meta")
                    return NotImplemented

            class ExampleTensor3(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device("meta")
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.prim.device'"
            with self.assertRaisesRegex(TypeError, err_msg):
                e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
                e.device()

            ten = torch.rand([1])
            e = ExampleTensor2(torch.randn(3, 3, device="cpu"), use_wrapper_subclass)
            self.assertEqual(e.device.type, "meta")
            self.assertEqual(ten.type_as(e).device.type, "meta")

            e = ExampleTensor3(torch.randn(3, 3, device="cpu"), use_wrapper_subclass)
            self.assertEqual(e.device.type, "meta")
            self.assertEqual(ten.type_as(e).device.type, "meta")

    def test_dim_slowpath(self):
        data = torch.randn(3, 3)

        for use_wrapper_subclass in [True, False]:

            class DimNotImplementedTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class DimImplementedTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.dim'"
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

    def test_sym_sizes_strides_slow_path(self):
        class TestTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls, (0,), dispatch_sizes_strides_policy="sizes"
                )
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func in (
                    torch.ops.aten.sym_size.default,
                    torch.ops.aten.sym_stride.default,
                ):
                    from torch._dynamo.source import ConstantSource
                    from torch.fx.experimental.symbolic_shapes import (
                        DimDynamic,
                        ShapeEnv,
                    )

                    shape_env = ShapeEnv()
                    si = shape_env.create_symintnode(
                        shape_env.create_symbol(
                            123,
                            source=ConstantSource("abc"),
                            dynamic_dim=DimDynamic.DUCK,
                            constraint_dim=None,
                        ),
                        hint=123,
                    )
                    return (si,)

        t = TestTensor()
        si = t.size()[0]
        self.assertIsInstance(si, torch.SymInt)
        si = t.stride()[0]
        self.assertIsInstance(si, torch.SymInt)

    def test_strides_slow_path(self):
        for use_wrapper_subclass in [True, False]:

            class StridesNotImplemented(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class StridesCustomReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func == torch.ops.aten.sym_stride.default:
                        return (4, 2)
                    return NotImplemented

            class StridesDefaultReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func == torch.ops.aten.sym_stride.default:
                        return None
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_stride'"
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
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    return NotImplemented

            class SizesCustomReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

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
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return None
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_size'"
            e = SizesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.size()

            e = SizesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.size(), (5, 3))

            e = SizesDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.size(), (4, 2))

    def test_custom_size_policy_dynamic_shapes(self):
        data = torch.randn(6, 2)

        class CustomSizeDynamicShapesTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, inner):
                return torch.Tensor._make_wrapper_subclass(
                    # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
                    # Calling the overload that has kwargs causes us to go down the first overload path,
                    # which will **always** specialize sizes.
                    # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
                    cls,
                    inner.size(),
                    inner.stride(),
                    None,
                    None,
                    inner.dtype,
                    inner.layout,
                    inner.device,
                    False,
                    inner.requires_grad,
                    "sizes",
                )

            def __init__(self, inner):
                self.inner = inner

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if func == torch.ops.aten.sym_size.default:
                    return args[0].inner.shape
                if func == torch.ops.aten.sym_stride.default:
                    return args[0].inner.shape
                return NotImplemented

        x = torch.ones(2, 2)

        def trace_fn(x):
            x_wrapper = CustomSizeDynamicShapesTensor(x)
            return x_wrapper.size(), x_wrapper.stride()

        fx_g = make_fx(trace_fn, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            fx_g.code.strip(),
            """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None
    return ((sym_size_int, sym_size_int_1), (sym_size_int, sym_size_int_1))""",
        )

    def test_data_ptr_respects_numel_slow_path(self):
        data = torch.randn(6, 2)

        class NumelDefaultReturn(torch.Tensor):
            @staticmethod
            def __new__(cls, data, wrapper):
                return TestPythonDispatch.subclass_helper(
                    cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                )

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if func.overloadpacket == torch.ops.aten.dim:
                    return data.dim()
                if func.overloadpacket == torch.ops.aten.numel:
                    numel_called[0] = True
                    return None
                return NotImplemented

        for use_wrapper_subclass in (False, True):
            numel_called = [False]
            e = NumelDefaultReturn(torch.randn(2, 2), use_wrapper_subclass)
            e.data_ptr()
            self.assertTrue(numel_called[0])

    def test_layout_slow_path(self):
        for use_wrapper_subclass in [True, False]:
            data = torch.randn(6, 2)

            class LayoutNotImplemented(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            class LayoutCustomReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.layout:
                        return torch.sparse_csr
                    return NotImplemented

            class LayoutDefaultReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func.overloadpacket == torch.ops.prim.layout:
                        return data.layout
                    return NotImplemented

            err_msg = "Multiple dispatch failed for 'torch.ops.prim.layout'"
            e = LayoutNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.layout

            e = LayoutCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.sparse_csr)

            e = LayoutDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.strided)

    def test_wrapper_subclass_reentrant_dispatch_with_mode(self):
        # Tests the interaction between a wrapper subclass using reentrant dispatch
        # and a TorchDispatchMode. See https://github.com/pytorch/pytorch/issues/136565

        # simple passthrough TorchDispatchMode
        class CustomDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=..., kwargs=None):
                return func(*args, **kwargs)

        # derive from TwoTensor to minimize boilerplate
        class MySubclass(TwoTensor):
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                with torch.overrides.enable_reentrant_dispatch():
                    return func(args[0].a)

        t = MySubclass(torch.rand(2), torch.rand(2))
        with CustomDispatchMode():
            res = t.clone()

        self.assertEqual(res, t.a)
        self.assertIs(type(res), torch.Tensor)

    def test_custom_dispatch_mode_supports_higher_order_operators(self):
        class Mode(TorchDispatchMode):
            supports_higher_order_operators = True

            def __torch_dispatch__(self, func, types, args=..., kwargs=None):
                if func is torch.ops.higher_order.cond:
                    return torch.ones(3, 3)
                return NotImplemented

        pred = torch.tensor(True)
        x = torch.randn(1, 1)
        with Mode():
            out = torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))
        self.assertEqual(out, torch.ones(3, 3))

    def test_custom_dispatch_mode_not_supports_higher_order_operators(self):
        class Mode(TorchDispatchMode):
            supports_higher_order_operators = False

            def __torch_dispatch__(self, func, types, args=..., kwargs=None):
                if func is torch.ops.higher_order.cond:
                    return torch.ones(3, 3)
                return NotImplemented

        pred = torch.tensor(True)
        x = torch.randn(1, 1)
        with self.assertRaisesRegex(
            NotImplementedError,
            "There was no rule registered for HigherOrderOperator cond and mode",
        ):
            with Mode():
                torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))

    def test_dispatch_uint64(self):
        class DummyMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args, kwargs):
                self.last_args = args
                return func(*args, **kwargs)

        # Value that could not be interpreted as signed int64
        uarg = 2**63 + 1
        with DummyMode() as m:
            a = torch.full((3, 3), uarg, dtype=torch.uint64)
            self.assertEqual(m.last_args[1], uarg)
        self.assertTrue((a == uarg).all().item())


class TestPythonDispatcher(TestCase):
    def test_basic(self):
        x = torch.randn(2, requires_grad=True)
        r = torch._C._EnablePythonDispatcher()
        torch.add(x, x)

    def test_lstsq(self):
        a = torch.randn(4, 3)
        b = torch.rand(4, 3)
        expected_shape = torch.linalg.lstsq(a, b).solution.shape
        r = torch._C._EnablePythonDispatcher()
        python_disp_shape = torch.linalg.lstsq(a, b).solution.shape
        self.assertEqual(expected_shape, python_disp_shape)


class TestWrapperSubclassAliasing(TestCase):
    def _test_wrapper_subclass_aliasing(self, op, args, kwargs):
        def to_subclass(t: torch.Tensor):
            return TwoTensor(t, t.clone())

        result_ref = op(*args, **kwargs)

        args_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, args)
        kwargs_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, kwargs)

        result_test = op(*args_subclass, **kwargs_subclass)

        args_ref_flat = pytree.arg_tree_leaves(*args, **kwargs)
        args_ref_flat_tensors = [
            x for x in args_ref_flat if isinstance(x, torch.Tensor)
        ]

        args_test_flat = pytree.tree_leaves((args_subclass, kwargs_subclass))
        args_test_flat_tensors = [
            x for x in args_test_flat if isinstance(x, torch.Tensor)
        ]

        result_ref_flat = pytree.tree_leaves(result_ref)
        result_ref_flat_tensors = [
            x for x in result_ref_flat if isinstance(x, torch.Tensor)
        ]

        result_test_flat = pytree.tree_leaves(result_test)
        result_test_flat_tensors = [
            x for x in result_test_flat if isinstance(x, torch.Tensor)
        ]

        for o_ref, o_test in zip(result_ref_flat_tensors, result_test_flat_tensors):
            for a_ref, a_test in zip(args_ref_flat_tensors, args_test_flat_tensors):
                out_is_inpt = o_ref is a_ref
                if out_is_inpt:
                    self.assertTrue(o_test is a_test)

                out_aliases_inpt = StorageWeakRef(
                    o_ref.untyped_storage()
                ) == StorageWeakRef(a_ref.untyped_storage())
                if out_aliases_inpt:
                    self.assertTrue(
                        StorageWeakRef(o_test.untyped_storage())
                        == StorageWeakRef(a_test.untyped_storage())
                    )
                else:
                    self.assertFalse(
                        StorageWeakRef(o_test.untyped_storage())
                        == StorageWeakRef(a_test.untyped_storage())
                    )

    # This tests the correctness of `torch.utils._python_dispatch.return_and_correct_aliasing`,
    # a util for wrapper subclasses to promise correct aliasing behavior.
    # It's probably overkill to test every OpInfo,
    # so I picked a sampling of ops with representative schemas.
    @ops(
        [
            op
            for op in op_db
            if op.name
            in [
                "mul",  # out-of-place
                "cat",  # out-of-place (TensorList input)
                "index",  # out-of-place (Optional TensorList input)
                "mul_",  # inplace
                "view",  # view
                "t_",  # inplace-view
                "split",  # view (multi-return)
                "native_batch_norm",  # mutable op (returns outputs and mutates some inputs)
            ]
        ],
        allowed_dtypes=(torch.float,),
    )
    def test_wrapper_subclass_aliasing(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)
        args = (sample.input, *sample.args)
        kwargs = sample.kwargs
        self._test_wrapper_subclass_aliasing(op, args, kwargs)

    @ops(custom_op_db, allowed_dtypes=(torch.float,))
    def test_wrapper_subclass_aliasing_custom(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)
        args = (sample.input, *sample.args)
        kwargs = sample.kwargs
        self._test_wrapper_subclass_aliasing(op, args, kwargs)

    def test_wrapper_subclass_aliasing_conv2d(self, device):
        args = (torch.randn(4, 4, 4, 4), torch.randn(4, 4, 4, 4))
        kwargs = {}
        # conv2d has a default arg 'int[2] strides=0',
        # which torchscript expands into 'int[2] strides=[0, 0]'
        # Make sure that _return_and_correct_aliasing can handle this case
        # (I'm using inference_mode to make sure conv2d doesn't decompose and goes to torch_dispatch)
        with torch.inference_mode():
            self._test_wrapper_subclass_aliasing(
                torch.ops.aten.conv2d.default, args, kwargs
            )

    def test_wrapper_subclass_aliasing_out_op(self, device):
        # Make sure that _return_and_correct_aliasing can handle kwargs w mutable tensors
        args = (torch.ones(4), torch.ones(4))
        kwargs = {"out": torch.empty(4)}
        self._test_wrapper_subclass_aliasing(torch.ops.aten.add.out, args, kwargs)

    def test_wrapper_subclass_aliasing_fft_fft2(self, device):
        args = (torch.randn(4, 4),)
        kwargs = {}
        # fft_fft2 has a default arg 'int[1] dim=[-2,-1]',
        # Make sure that _return_and_correct_aliasing can handle this case
        # (I'm using inference_mode to make sure fft_fft2 doesn't decompose and goes to torch_dispatch)
        with torch.inference_mode():
            self._test_wrapper_subclass_aliasing(torch.ops.aten.fft_fft2, args, kwargs)


instantiate_device_type_tests(TestWrapperSubclassAliasing, globals())

if __name__ == "__main__":
    run_tests()
