# Owner(s): ["module: __torch_dispatch__"]

import tempfile
import torch
from copy import deepcopy
from torch.library import Library, impl
from torch.fx.experimental.proxy_tensor import ShapeEnv
from torch import SymInt
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda.jiterator import _create_jit_fn
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, IS_WINDOWS, TEST_CUDA
from torch.utils._mode_utils import no_dispatch, all_same_mode
from torch.testing._internal.logging_tensor import LoggingTensor, LoggingTensorReentrant, LoggingTensorMode, \
    log_input, capture_logs, capture_logs_with_logging_tensor_mode
from torch.utils._pytree import tree_map, tree_map_only
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode, _get_current_dispatch_mode_stack
from torch._custom_op import custom_op, CustomOp
from torch.fx.experimental.proxy_tensor import make_fx
import typing
import collections
from typing import Optional, Tuple, Union, List, Callable, Sequence
from torch import Tensor
import itertools

import logging
import sys
import torch._dynamo
import torch.testing._internal.custom_op_db
import re


class TestDispatcherPythonBindings(TestCase):
    def test_call_boxed(self) -> None:
        sin = torch._C._dispatch_find_schema_or_throw("aten::sin", "")
        x = torch.randn(3)
        y = torch._C._dispatch_call_boxed(sin, x)
        self.assertEqual(y, x.sin())


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
            return args[0].clone()

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
            return args[0].clone()
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
            return args[0].clone()

        x = torch.tensor([1, 2])
        self.assertEqual(torch.ops.foo.sum(x), x)

        my_lib2 = Library("foo", "IMPL")

        # Example 2
        @torch.library.impl(my_lib2, torch.ops.foo.sum.default, "ZeroTensor")
        def my_sum_zt(*args, **kwargs):
            if args[0]._is_zerotensor():
                return torch._efficientzerotensor(args[0].shape)
            else:
                return args[0].clone()

        y = torch._efficientzerotensor(3)
        self.assertTrue(torch.ops.foo.sum(y)._is_zerotensor())
        self.assertEqual(torch.ops.foo.sum(x), x)

        del my_lib2
        del my_lib1

    def test_create_new_library_fragment_no_existing(self):
        my_lib = Library("foo", "FRAGMENT")

        my_lib.define("sum2(Tensor self) -> Tensor")

        @torch.library.impl(my_lib, "sum2", "CPU")
        def my_sum(*args, **kwargs):
            return args[0]

        x = torch.tensor([1, 2])
        self.assertEqual(torch.ops.foo.sum2(x), x)

        del my_lib

    def test_create_new_library_fragment_with_existing(self):
        my_lib1 = Library("foo", "DEF")

        # Create a fragment
        my_lib2 = Library("foo", "FRAGMENT")

        my_lib2.define("sum4(Tensor self) -> Tensor")

        @torch.library.impl(my_lib2, "sum4", "CPU")
        def my_sum4(*args, **kwargs):
            return args[0]

        x = torch.tensor([1, 2])
        self.assertEqual(torch.ops.foo.sum4(x), x)

        # Create another fragment
        my_lib3 = Library("foo", "FRAGMENT")

        my_lib3.define("sum3(Tensor self) -> Tensor")

        @torch.library.impl(my_lib3, "sum3", "CPU")
        def my_sum3(*args, **kwargs):
            return args[0]

        x = torch.tensor([1, 2])
        self.assertEqual(torch.ops.foo.sum3(x), x)

        del my_lib1
        del my_lib2
        del my_lib3

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

        for kind in ('DEF', 'FRAGMENT'):
            with self.assertRaisesRegex(ValueError, "reserved namespace"):
                my_lib1 = Library("prim", kind)

    def test_returning_symint(self) -> None:
        shape_env = ShapeEnv()
        fake_tensor_mode = FakeTensorMode(shape_env=shape_env)

        ft = fake_tensor_mode.from_tensor(torch.rand(2, 3))

        s0, s1 = ft.shape

        tlib = Library("tlib", "DEF")
        tlib.define("sqsum(SymInt a, SymInt b) -> SymInt")

        @impl(tlib, "sqsum", "CompositeExplicitAutograd")
        def sqsum(a: SymInt, b: SymInt):
            return a * a + b * b

        out = torch.ops.tlib.sqsum.default(s0, s1)
        out_val = shape_env.evaluate_expr(out.node.expr)
        self.assertEquals(out_val, 13)


class TestCustomOp(TestCase):
    test_ns = '_test_custom_op'

    def tearDown(self):
        import torch._custom_op
        keys = list(torch._custom_op.global_registry.keys())
        for key in keys:
            if not key.startswith(f'{TestCustomOp.test_ns}::'):
                continue
            torch._custom_op.global_registry[key]._destroy()

    def test_invalid_schemas(self):
        # function schmea validation goes through torchgen, so this is just a
        # basic test.
        with self.assertRaisesRegex(AssertionError, 'Invalid function schema: foo'):
            @custom_op(f'{TestCustomOp.test_ns}::foo', "(")
            def foo(x):
                ...

    def test_name_must_match(self):
        with self.assertRaisesRegex(ValueError, 'to have name'):
            @custom_op(f'{TestCustomOp.test_ns}::foo', "(Tensor x) -> Tensor")
            def bar(x):
                ...

        with self.assertRaisesRegex(ValueError, 'to have name'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def baz(x: Tensor) -> Tensor:
                ...

    def test_unsupported_schemas(self):
        def foo(x):
            ...

        with self.assertRaisesRegex(ValueError, 'does not support non-functional'):
            custom_op(f'{TestCustomOp.test_ns}::foo', '(Tensor(a!) x) -> Tensor(a)')(foo)
        with self.assertRaisesRegex(ValueError, 'does not support view functions'):
            custom_op(f'{TestCustomOp.test_ns}::foo', '(Tensor(a) x) -> Tensor(a)')(foo)
        with self.assertRaisesRegex(ValueError, 'no outputs'):
            custom_op(f'{TestCustomOp.test_ns}::foo', '(Tensor x) -> ()')(foo)
        with self.assertRaisesRegex(ValueError, 'self'):
            custom_op(f'{TestCustomOp.test_ns}::foo', '(Tensor self) -> ()')(foo)

    def test_schema_matches_signature(self):
        with self.assertRaisesRegex(ValueError, 'signature to match'):
            @custom_op(f'{TestCustomOp.test_ns}::blah', '(Tensor y) -> Tensor')
            def blah(x):
                pass

        with self.assertRaisesRegex(ValueError, 'signature to match'):
            @custom_op(f'{TestCustomOp.test_ns}::blah2', '(Tensor x, *, Tensor y) -> Tensor')
            def blah2(x, y):
                pass

        with self.assertRaisesRegex(ValueError, 'signature to match'):
            @custom_op(f'{TestCustomOp.test_ns}::blah3', '(Tensor x, *, Tensor w, Tensor z) -> Tensor')
            def blah3(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, 'signature to match'):
            @custom_op(f'{TestCustomOp.test_ns}::blah4', '(Tensor x, *, Tensor z, Tensor y) -> Tensor')
            def blah4(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, 'not supported'):
            @custom_op(f'{TestCustomOp.test_ns}::blah5', '(Tensor x) -> Tensor')
            def blah5(*args):
                pass

        with self.assertRaisesRegex(ValueError, 'not supported'):
            @custom_op(f'{TestCustomOp.test_ns}::blah6', '(*, Tensor z, Tensor y) -> Tensor')
            def blah6(**kwargs):
                pass

        with self.assertRaisesRegex(ValueError, 'default arguments'):
            @custom_op(f'{TestCustomOp.test_ns}::blah7', '(Tensor x, *, Tensor y) -> Tensor')
            def blah7(x=1, *, y):
                pass

        with self.assertRaisesRegex(ValueError, 'default arguments'):
            @custom_op(f'{TestCustomOp.test_ns}::blah8', '(Tensor x, *, Tensor y) -> Tensor')
            def blah8(x, *, y=1):
                pass

        # kwonly-arg works
        @custom_op(f'{TestCustomOp.test_ns}::blah9', '(Tensor x, *, Tensor y) -> Tensor')
        def blah9(x, *, y):
            pass

    def test_unsupported_annotation_categories(self):
        with self.assertRaisesRegex(ValueError, 'varargs'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(*args):
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'varkwargs'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(**kwargs):
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'must have a type annotation'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x):
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'default value'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Optional[Tensor] = None):
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'default value'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Optional[Tensor] = None):
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'either Tensor or a Tuple'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor) -> int:
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'either Tensor or a Tuple'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor) -> Tuple[Tensor, int]:
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'either Tensor or a Tuple'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor) -> Tuple[Tensor, ...]:
                ...
            del foo

    def test_supported_param_types(self):
        def generate_examples(typ):
            if typ is int:
                return [17]
            if typ is float:
                return [3.14]
            if typ is bool:
                return [True]
            if typ is str:
                return ["foo"]
            if typ is torch.dtype:
                return [torch.float32]
            if typ is torch.device:
                return [torch.device('cpu')]
            if typ == torch.types.Number:
                return [2.718]
            if typ is torch.Tensor:
                return [torch.tensor(3)]
            if typ == Optional[torch.types.Number]:
                return [None, 2.718]
            origin = typing.get_origin(typ)
            if origin is Union:
                args = typing.get_args(typ)
                assert len(args) == 2 and (args[0] is type(None) or args[1] is type(None))
                elt = args[0] if args[1] is type(None) else args[1]
                return generate_examples(elt) + [None]
            if origin is collections.abc.Sequence:
                args = typing.get_args(typ)
                assert len(args) == 1
                examples = generate_examples(args[0])
                return list(itertools.product(examples, examples)) + []
            raise AssertionError(f"unsupported param type {typ}")

        for typ in torch._custom_op.SUPPORTED_PARAM_TYPES:
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor, y: typ) -> Tensor:
                ...

            yeet = None

            @foo.impl(['cpu'])
            def foo_cpu(x, y):
                nonlocal yeet
                yeet = y
                return x.clone()

            try:
                for example in generate_examples(typ):
                    foo(torch.randn([]), example)
                    self.assertEqual(yeet, example, msg=f'{typ} {example}')
                    yeet = None
            finally:
                foo._destroy()
                del foo
                del foo_cpu

    def test_sequences(self):
        # Sequence[int] gets automagically turned into int[] in the schema.
        # This test checks that we actually do support arbitrary sequence types.
        class MySequence(collections.abc.Sequence):
            def __init__(self):
                self._container = [1, 2, 3]

            def __getitem__(self, idx):
                return self._container[idx]

            def __len__(self):
                return len(self._container)

        @custom_op("blah::foo")
        def foo(x: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
            ...

        called = 0

        @foo.impl('cpu')
        def foo_cpu(x, sizes):
            nonlocal called
            called += 1
            # Dispatcher will normalize the sequence type into a List
            self.assertEqual(sizes, [1, 2, 3])
            return x.clone()

        x = torch.randn([])
        seq = MySequence()
        foo(x, seq)
        self.assertEqual(called, 1)

    def test_unsupported_param_types(self):
        # Not comprehensive (it doesn't need to be), just a check that our mechanism works
        with self.assertRaisesRegex(ValueError, 'unsupported type'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor, y: List[Optional[int]]) -> Tensor:
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'unsupported type'):
            # int[N] in Dispatcher is a bit wild, so we don't try to support it.
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'unsupported type'):
            # We could theoretically support this, but the syntax for suporting
            # int[] is Sequence[int]
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor, y: List[int]) -> Tensor:
                ...
            del foo

        with self.assertRaisesRegex(ValueError, 'unsupported type'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: Tensor, y: Callable) -> Tensor:
                ...
            del foo

    def test_custom_op_behaves_like_function(self):
        from torch.testing._internal.custom_op_db import numpy_mul
        self.assertEqual(numpy_mul.__name__, 'numpy_mul')
        self.assertEqual(numpy_mul.__module__, 'torch.testing._internal.custom_op_db')
        self.assertTrue(callable(numpy_mul))

    def test_custom_op_repr(self):
        from torch.testing._internal.custom_op_db import numpy_mul
        expected = '<CustomOp(op="_torch_testing::numpy_mul")>'
        self.assertEqual(repr(numpy_mul), expected)

    def test_supported_schemas(self):
        # All of these should already be tested by PyTorch codegen
        # (we share the same mechanism), but here's a sanity check.
        schemas = [
            '(Tensor x) -> Tensor',
            '(Tensor x) -> Tensor y',
            '(Tensor[] x) -> Tensor y',
            '(Tensor x) -> (Tensor, Tensor)',
            '(Tensor x) -> (Tensor y, Tensor z)',
            '(Tensor x) -> (Tensor y, Tensor z)',
        ]
        other_schemas = [
            '(Tensor x, Tensor w) -> (Tensor y, Tensor z)',
            '(Tensor x, Tensor w) -> (Tensor, Tensor)',
            '(Tensor x, Tensor w) -> Tensor',
            '(Tensor? x, Tensor w) -> Tensor',
            '(Tensor? x, Tensor[] w) -> Tensor',
            '(Tensor x, int[] w) -> Tensor',
            '(Tensor x, SymInt[] w) -> Tensor',
            '(Tensor x, Scalar w) -> Tensor',
            '(Tensor x, float w) -> Tensor',
            '(Tensor x, float? w) -> Tensor',
            '(Tensor x, bool[] w) -> Tensor',
        ]

        def foo(x):
            ...

        def bar(x, w):
            ...

        for schema in schemas:
            op = custom_op(f'{TestCustomOp.test_ns}::foo', schema)(foo)
            op._destroy()
        for schema in other_schemas:
            op = custom_op(f'{TestCustomOp.test_ns}::bar', schema)(bar)
            op._destroy()

    def test_reserved_ns(self):
        from torch._custom_op import RESERVED_NS

        for ns in RESERVED_NS:
            with self.assertRaisesRegex(ValueError, 'is a reserved namespace'):
                @custom_op(f'{ns}::foo', '(Tensor x) -> Tensor')
                def foo(x):
                    ...
            with self.assertRaisesRegex(ValueError, 'is a reserved namespace'):
                @custom_op(f'{ns}::foo2')
                def foo2(x: torch.Tensor) -> torch.Tensor:
                    ...

    def test_private_ctor(self):
        with self.assertRaisesRegex(RuntimeError, 'CustomOp constructor is private'):
            CustomOp(None, None, None, None)

    def test_lifetime(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        # 3 references:
        # - foo (in this function)
        # - arg passed to sys.getrefcount
        # - global_registry
        self.assertEqual(sys.getrefcount(foo), 3)

        # We can't define an op multiple times,
        with self.assertRaisesRegex(RuntimeError, 'multiple times'):
            @custom_op(f'{TestCustomOp.test_ns}::foo')
            def foo(x: torch.Tensor) -> torch.Tensor:
                ...

        # Unless we delete the original op.
        foo._destroy()

        # Smoke test
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        foo._destroy()

    def test_autograd_notimplemented(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, 'Autograd has not been implemented'):
            foo(x)
        foo._destroy()

        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: Sequence[torch.Tensor]) -> torch.Tensor:
            ...

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, 'Autograd has not been implemented'):
            foo([y, x])
        foo._destroy()

        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            ...

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, 'Autograd has not been implemented'):
            foo(y, x)
        foo._destroy()

    def test_impl_cpu(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl('cpu')
        def foo_cpu(x):
            return x.sin()

        x = torch.randn(3)
        result = foo(x)
        self.assertEqual(result, foo_cpu(x))

    def test_impl_invalid_devices(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        def foo_impl(x):
            return x.sin()

        from torch._custom_op import SUPPORTED_DEVICE_TYPE_TO_KEY

        for device_type in SUPPORTED_DEVICE_TYPE_TO_KEY.keys():
            # Smoke test: should not raise error
            foo.impl(device_type)(foo_impl)

        # Not supported by this API: we can either support them in the future
        # or provide some other CustomOp.def_* function. This depends on how
        # common the use cases are.
        for invalid_type in ['hip', 'xla', 'mkldnn', ['cpu', 'hip']]:
            with self.assertRaisesRegex(ValueError, "we only support device_type"):
                foo.impl(invalid_type)(foo_impl)
        foo._destroy()

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_impl_separate(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl('cpu')
        def foo_cpu(x):
            return x.sin()

        @foo.impl('cuda')
        def foo_cuda(x):
            return x.cos()

        x = torch.randn(3)
        result = foo(x)
        self.assertEqual(result, foo_cpu(x))

        x_cuda = x.cuda()
        result = foo(x_cuda)
        self.assertEqual(result, foo_cuda(x_cuda))
        foo._destroy()

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_impl_multiple(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(['cpu', 'cuda'])
        def foo_impl(x):
            return x.cos()

        x = torch.randn(3)
        result = foo(x)
        self.assertEqual(result, foo_impl(x))

        x_cuda = x.cuda()
        result = foo(x_cuda)
        self.assertEqual(result, foo_impl(x_cuda))
        foo._destroy()

    def test_impl_meta(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x, dim):
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        x = torch.randn(2, 3, device='meta')
        result = foo(x, 1)
        self.assertEqual(result.shape, foo_meta(x, 1).shape)
        foo._destroy()

    def test_duplicate_impl(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x, dim):
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        with self.assertRaisesRegex(
                RuntimeError,
                r"already has a abstract impl.*at .*test_python_dispatch.py:\d+"):
            @foo.impl_abstract()
            def foo_meta2(x, dim):
                output_shape = list(x.shape)
                del output_shape[dim]
                return x.new_empty(output_shape)

        foo._destroy()

    def test_new_data_dependent_symint(self):
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x):
            ctx = torch._custom_op.get_ctx()
            with self.assertRaisesRegex(ValueError, "greater than or equal to 2"):
                ctx.create_unbacked_symint(min=1)
            with self.assertRaisesRegex(ValueError, "greater than or equal to 2"):
                ctx.create_unbacked_symint(min=-1)
            with self.assertRaisesRegex(ValueError, "SymInt"):
                ctx.create_unbacked_symint(max=x.numel())
            return torch.clone(x)

        x = torch.randn(2, 3, device='cpu')
        make_fx(foo, tracing_mode='symbolic')(x)
        foo._destroy()

    def test_meta_for_data_dependent_shape_operation(self):
        from torch.testing._internal.custom_op_db import numpy_nonzero

        x = torch.randn(10, device='meta')
        with self.assertRaisesRegex(RuntimeError, 'data-dependent output shape'):
            numpy_nonzero(x)

    def test_basic_make_fx(self):
        # More serious tests are in our CustomOp opinfo db,
        # this one is just a sanity check.
        @custom_op(f'{TestCustomOp.test_ns}::foo')
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x):
            return x.sum()

        x = torch.randn(3)
        gm = make_fx(foo, tracing_mode='symbolic')(x)
        self.assertTrue(f'{TestCustomOp.test_ns}.foo' in gm.code)
        foo._destroy()

    def test_abstract_registration_location(self):
        loc = torch.testing._internal.custom_op_db.numpy_nonzero._get_impl('abstract').location
        matches = re.match(r'.*custom_op_db.py:\d+', loc)
        self.assertIsNotNone(matches)

    def test_data_dependent_basic(self):
        from torch.testing._internal.custom_op_db import numpy_nonzero

        def f(x):
            return numpy_nonzero(x)

        x = torch.randn(5, 5)
        gm = make_fx(f, tracing_mode='symbolic')(x)
        self.assertTrue('nonzero' in gm.code)

    def test_data_dependent_fake_tracing(self):
        from torch.testing._internal.custom_op_db import numpy_nonzero

        def f(x):
            return numpy_nonzero(x)

        x = torch.randn(5, 5)
        with self.assertRaises(torch._subclasses.fake_tensor.DynamicOutputShapeException):
            make_fx(f, tracing_mode='fake')(x)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work on windows")
    def test_data_dependent_compile(self):
        import torch._dynamo.testing
        from torch._dynamo.utils import counters
        counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            return torch.ops._torch_testing.numpy_nonzero(x.clone()).clone()

        f(torch.randn(10))

        self.assertEqual(
            dict(counters['graph_break']),
            {'dynamic shape operator: _torch_testing.numpy_nonzero.default': 1},
        )

    # pre-existing problem: torch.compile(dynamic=True) will, by default,
    # graph break on data-dependent operations. Eventually we'll make it so
    # that it never graph breaks on data-dependent operations.
    @unittest.expectedFailure
    def test_data_dependent_nms_dynamic_compile(self):
        import torch._dynamo.testing
        from torch._dynamo.utils import counters
        counters.clear()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, dynamic=True)
        def f(x, s, i):
            return torch.ops._torch_testing.numpy_nms(x.clone(), s, i).clone()

        f(torch.randn(20, 4), torch.randn(20), 0.1)

        self.assertEqual(len(counters['graph_break']), 0)


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

    def test_optional_tensor_list(self) -> None:
        def weird(xs):
            print("woof")
            return torch.empty(())

        my_lib = Library("my_lib", "DEF")
        my_lib.define("weird(Tensor?[] self) -> Tensor")
        my_lib.impl("weird", weird, "CPU")
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(2, 2))
            log_input("x", x)
            torch.ops.my_lib.weird.default([None, x])

        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.my_lib.weird.default([None, LoggingTensor(tensor([[1., 1.],
        [1., 1.]]))])''')

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
        self.assertRaisesRegex(
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

    def test_torch_dispatch_mode_basic(self) -> None:
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                torch.empty([])
        self.assertExpectedInline('\n'.join(logs), """\
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)""")

    def test_torch_dispatch_mode_unrelated_tensors(self) -> None:
        x = torch.randn([])
        y = torch.randn([])
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                x + y
        self.assertExpectedInline('\n'.join(logs), """\
$2 = torch._ops.aten.add.Tensor($0, $1)""")

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

    def test_exception_handling(self):
        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class AMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func.__name__ == 'randn.default':
                    raise RuntimeError()
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
                raise ErrorA()

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

        x = torch.tensor(4.)
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

    def test_nesting_same_mode(self):
        # If the pushed mode is the same instance as the current mode, we allow pushing an already active mode.

        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode() as reenabled:
                with reenabled:
                    torch.empty([])
            self.assertExpectedInline('\n'.join(logs), """\
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0 = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)""")


    def test_error_using_class_method_on_mode(self):
        class A(TorchDispatchMode):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return func(args, kwargs)

        x = torch.tensor(5.)
        with self.assertRaisesRegex(RuntimeError, "classmethod is not supported, please make it a plain method"):
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

    def test_tolist_numpy_with_torch_dispatch_mode(self) -> None:
        x = LoggingTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(RuntimeError, "is not supported for tensor subclasses."):
            x.tolist()
        with self.assertRaisesRegex(RuntimeError, "is not supported for tensor subclasses."):
            x.numpy()
        with self.assertRaises(AssertionError):
            self.assertEqual(x, None)

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

        t = torch.tensor(5.)
        s = torch.Stream(stream_id=1, device_index=2, device_type=3)
        with TestMode(self):
            t.record_stream(s)

    def test_return_stream(self) -> None:
        l_def = torch.library.Library("test_return_stream", "DEF")
        l_def.define("return_stream(Tensor self) -> Stream")
        l_impl = torch.library.Library("test_return_stream", "IMPL", "CPU")
        l_impl.impl("return_stream", lambda _: torch.Stream(stream_id=0, device_index=1, device_type=2))

        class TestMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return torch.Stream(stream_id=1, device_index=2, device_type=3)

        t = torch.tensor(5.)
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

    def test_subclass_autograd_device_check(self) -> None:
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

    def test_fancy_strides(self):
        calls = []

        class ExampleTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return TestPythonDispatch.subclass_helper(cls, data, False, dispatch_sizes_strides_policy="strides")

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if func in [
                    torch.ops.aten.is_contiguous.default,
                    torch.ops.aten.is_contiguous.memory_format,
                    torch.ops.aten.is_strides_like_format.default,
                    torch.ops.aten.is_non_overlapping_and_dense.default,
                    torch.ops.aten.stride.default
                ]:
                    calls.append((func, list(args)[1:]))
                    return None
                with no_dispatch():
                    return func(*args, **kwargs)

        e = ExampleTensor(torch.randn(2, 2))
        self.assertFalse(e.is_contiguous(memory_format=torch.channels_last))
        self.assertEqual(calls, [(torch.ops.aten.is_contiguous.memory_format, [torch.channels_last])])
        calls.clear()
        self.assertFalse(torch.ops.aten.is_strides_like_format.default(e, torch.channels_last))
        self.assertEqual(calls, [(torch.ops.aten.is_strides_like_format.default, [torch.channels_last])])
        calls.clear()
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(e))
        self.assertEqual(calls, [(torch.ops.aten.is_non_overlapping_and_dense.default, [])])

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
                    if func == torch.ops.aten.sym_stride.default:
                        return (4, 2)
                    return NotImplemented

            class StridesDefaultReturn(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(cls, data, wrapper, dispatch_sizes_strides_policy="strides")

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    if func == torch.ops.aten.sym_stride.default:
                        return None
                    return NotImplemented

            err_msg = "no implementation found for 'torch.ops.aten.sym_stride'"
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
            with self.assertRaisesRegex(TypeError, err_msg):
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

if __name__ == '__main__':
    run_tests()
