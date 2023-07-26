# Owner(s): ["module: custom-operators"]

from torch.testing._internal.common_utils import *  # noqa: F403
from torch.testing._internal.common_device_type import *  # noqa: F403
import collections

import itertools
import re
import typing

import torch.testing._internal.custom_op_db
from functorch import make_fx
from torch import Tensor
from torch._custom_op.impl import custom_op, CustomOp
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.optests.compile_check import operator_compile_check
from typing import *  # noqa: F403


@unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
class TestCustomOpTesting(TestCase):
    def setUp(self):
        self.test_ns = "_test_custom_op"
        self.libraries = []

    def tearDown(self):
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{self.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, self.test_ns):
            del torch.ops._test_custom_op
        for lib in self.libraries:
            del lib.m
        del self.libraries

    def ns(self):
        return getattr(torch.ops, self.test_ns)

    def lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")
        self.libraries.append(result)
        return result

    def test_incorrect_schema_mutation(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                guard = torch._C._AutoDispatchBelowAutograd()
                try:
                    return op(x)
                finally:
                    del guard

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            x.sin_()
            return x.clone()

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")

        def f(x):
            x = x.clone()
            v = x.view_as(x)
            y = op(v)
            return x

        x = torch.tensor(3.14159 / 3, requires_grad=True, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Argument x is not defined as mutable but was mutated"
        ):
            operator_compile_check(f, (x,), {})

    def test_incorrect_schema_view(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                with torch._C._AutoDispatchBelowAutograd():
                    with torch._C._ExcludeDispatchKeyGuard(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                    ):
                        return op(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            return x.view_as(x)

        def foo_meta(x):
            return x.view_as(x)

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_meta, "Meta")

        def f(x):
            x = x.clone()
            y = op(x)
            x.sin_()
            return y

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "Argument x is not defined to alias output but was aliasing"
        ):
            operator_compile_check(f, (x,), {})

    def test_missing_abstract_impl(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                return 2 * gx

        def foo_impl(x):
            return torch.tensor(x.cpu().numpy() ** 2, device=x.device)

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")

        def f(x):
            y = op(x)
            return y.sum(0)

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(
            torch._subclasses.fake_tensor.UnsupportedOperatorException,
            "_test_custom_op.foo.default",
        ):
            operator_compile_check(f, (x,), {})

    def test_incorrect_abstract_impl(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                guard = torch._C._AutoDispatchBelowAutograd()
                guard2 = torch._C.ExcludeDispatchKeyGuard(
                    torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                )
                try:
                    return op(x)
                finally:
                    del guard
                    del guard2

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            return x**2

        def foo_meta(x):
            return x.unsqueeze(1) ** 2

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_meta, "Meta")

        def f(x):
            y = op(x)
            return y.sum(0)

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "Shapes .* are not equal"):
            operator_compile_check(f, (x,), {})

    def test_missing_functionalization(self, device):
        lib = self.lib()
        lib.define("foo(Tensor(a!) x) -> Tensor(a!)")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            return x.sin_()

        def foo_meta(x):
            return x

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_meta, "Meta")

        def f(x):
            x = x.clone()
            y = op(x)
            return y.sum(0)

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Getting these operators to work with functionalization requires some extra work",
        ):
            operator_compile_check(f, (x,), {})

    def test_autograd_registered_at_backend(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gx):
                return gx * 0.5

        lib.impl("foo", Foo.apply, "CPU")
        lib.impl("foo", Foo.apply, "CUDA")
        lib.impl("foo", lambda x: x.clone(), "Meta")

        def f(x):
            y = op(x)
            return x + y

        x = torch.randn([], requires_grad=True)

        with self.assertRaisesRegex(AssertionError, "mismatched requires_grad-ness"):
            operator_compile_check(f, (x,), {})

        # I'm not sure why this is necessary
        del lib

    def test_global_state_mutation(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            invoked = 0

            @staticmethod
            def forward(ctx, x):
                Foo.invoked += 1
                return x.clone() * Foo.invoked

            @staticmethod
            def backward(ctx, gx):
                return gx

        lib.impl("foo", Foo.apply, "CompositeImplicitAutograd")

        def f(x):
            return op(x)

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(AssertionError, "not completely traceable"):
            operator_compile_check(f, (x,), {})

    @ops(custom_op_db, dtypes=OpDTypes.any_one)
    def test_operator_compile_check_op(self, device, dtype, op):
        for sample_input in op.sample_inputs(
            device, dtype, requires_grad=op.supports_autograd
        ):
            dynamic_only = op.name in ("NumpyNMSCustomOp", "NumpyNonzeroCustomOp")
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            operator_compile_check(
                op.op,
                args,
                kwargs,
                supports_autograd=op.supports_autograd,
                dynamic_only=dynamic_only,
                fullgraph=False,  # Dynamo graph breaks on CustomOp today
            )

    def test_operator_compile_check_fails_basic(self, device):
        @custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sum()

        x = torch.randn(3, device=device, requires_grad=True)
        # Triggers the CustomOp autograd NYI error
        with self.assertRaisesRegex(
            RuntimeError, "Autograd has not been implemented for operator"
        ):
            operator_compile_check(lambda x: foo(x), (x,), {})

    def test_assert_raises_regex(self, device):
        from torch.testing._internal.optests.aot_autograd import assert_raises_regex

        with assert_raises_regex(RuntimeError, "c"):
            raise RuntimeError("abcd")
        with assert_raises_regex(RuntimeError, "c.*"):
            raise RuntimeError("abcd")
        with self.assertRaisesRegex(AssertionError, "instead got"):
            with assert_raises_regex(RuntimeError, "c.*"):
                raise ValueError("abcd")
        with self.assertRaisesRegex(AssertionError, "Expected exception"):
            with assert_raises_regex(RuntimeError, "c.*"):
                pass
        with self.assertRaisesRegex(AssertionError, "to match regex"):
            with assert_raises_regex(RuntimeError, "f"):
                raise RuntimeError("abcd")


class TestCustomOp(TestCase):
    test_ns = "_test_custom_op"

    def tearDown(self):
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{TestCustomOp.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()

    def test_invalid_schemas(self):
        # function schmea validation goes through torchgen, so this is just a
        # basic test.
        with self.assertRaisesRegex(AssertionError, "Invalid function schema: foo"):

            @custom_op(f"{TestCustomOp.test_ns}::foo", "(")
            def foo(x):
                ...

    def test_name_must_match(self):
        with self.assertRaisesRegex(ValueError, "to have name"):

            @custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor x) -> Tensor")
            def bar(x):
                ...

        with self.assertRaisesRegex(ValueError, "to have name"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def baz(x: Tensor) -> Tensor:
                ...

    def test_unsupported_schemas(self):
        def foo(x):
            ...

        with self.assertRaisesRegex(ValueError, "does not support non-functional"):
            custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor(a!) x) -> Tensor(a)")(
                foo
            )
        with self.assertRaisesRegex(ValueError, "does not support view functions"):
            custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor(a) x) -> Tensor(a)")(foo)
        with self.assertRaisesRegex(ValueError, "no outputs"):
            custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor x) -> ()")(foo)
        with self.assertRaisesRegex(ValueError, "self"):
            custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor self) -> ()")(foo)

    def test_schema_matches_signature(self):
        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(f"{TestCustomOp.test_ns}::blah", "(Tensor y) -> Tensor")
            def blah(x):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah2", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah2(x, y):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah3",
                "(Tensor x, *, Tensor w, Tensor z) -> Tensor",
            )
            def blah3(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah4",
                "(Tensor x, *, Tensor z, Tensor y) -> Tensor",
            )
            def blah4(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):

            @custom_op(f"{TestCustomOp.test_ns}::blah5", "(Tensor x) -> Tensor")
            def blah5(*args):
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah6", "(*, Tensor z, Tensor y) -> Tensor"
            )
            def blah6(**kwargs):
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah7", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah7(x=1, *, y):
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah8", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah8(x, *, y=1):
                pass

        # kwonly-arg works
        @custom_op(
            f"{TestCustomOp.test_ns}::blah9", "(Tensor x, *, Tensor y) -> Tensor"
        )
        def blah9(x, *, y):
            pass

    def test_unsupported_annotation_categories(self):
        with self.assertRaisesRegex(ValueError, "varargs"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(*args):
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "varkwargs"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(**kwargs):
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "must have a type annotation"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x):
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "default value"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Optional[Tensor] = None):
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "default value"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Optional[Tensor] = None):
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "either Tensor or a Tuple"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor) -> int:
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "either Tensor or a Tuple"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor) -> Tuple[Tensor, int]:
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "either Tensor or a Tuple"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
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
                return [torch.device("cpu")]
            if typ == torch.types.Number:
                return [2.718]
            if typ is torch.Tensor:
                return [torch.tensor(3)]
            if typ == Optional[torch.types.Number]:
                return [None, 2.718]
            origin = typing.get_origin(typ)
            if origin is Union:
                args = typing.get_args(typ)
                assert len(args) == 2 and (
                    args[0] is type(None) or args[1] is type(None)
                )
                elt = args[0] if args[1] is type(None) else args[1]
                return generate_examples(elt) + [None]
            if origin is collections.abc.Sequence:
                args = typing.get_args(typ)
                assert len(args) == 1
                examples = generate_examples(args[0])
                return list(itertools.product(examples, examples)) + []
            raise AssertionError(f"unsupported param type {typ}")

        for typ in torch._custom_op.impl.SUPPORTED_PARAM_TYPES:

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: typ) -> Tensor:
                ...

            yeet = None

            @foo.impl(["cpu"])
            def foo_cpu(x, y):
                nonlocal yeet
                yeet = y
                return x.clone()

            try:
                for example in generate_examples(typ):
                    foo(torch.randn([]), example)
                    self.assertEqual(yeet, example, msg=f"{typ} {example}")
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

        @foo.impl("cpu")
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
        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: List[Optional[int]]) -> Tensor:
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # int[N] in Dispatcher is a bit wild, so we don't try to support it.
            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # We could theoretically support this, but the syntax for suporting
            # int[] is Sequence[int]
            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: List[int]) -> Tensor:
                ...

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Callable) -> Tensor:
                ...

            del foo

    def test_custom_op_behaves_like_function(self):
        from torch.testing._internal.custom_op_db import numpy_mul

        self.assertEqual(numpy_mul.__name__, "numpy_mul")
        self.assertEqual(numpy_mul.__module__, "torch.testing._internal.custom_op_db")
        self.assertTrue(callable(numpy_mul))

    def test_custom_op_repr(self):
        from torch.testing._internal.custom_op_db import numpy_mul

        expected = '<CustomOp(op="_torch_testing::numpy_mul")>'
        self.assertEqual(repr(numpy_mul), expected)

    def test_supported_schemas(self):
        # All of these should already be tested by PyTorch codegen
        # (we share the same mechanism), but here's a sanity check.
        schemas = [
            "(Tensor x) -> Tensor",
            "(Tensor x) -> Tensor y",
            "(Tensor[] x) -> Tensor y",
            "(Tensor x) -> (Tensor, Tensor)",
            "(Tensor x) -> (Tensor y, Tensor z)",
            "(Tensor x) -> (Tensor y, Tensor z)",
        ]
        other_schemas = [
            "(Tensor x, Tensor w) -> (Tensor y, Tensor z)",
            "(Tensor x, Tensor w) -> (Tensor, Tensor)",
            "(Tensor x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor[] w) -> Tensor",
            "(Tensor x, int[] w) -> Tensor",
            "(Tensor x, SymInt[] w) -> Tensor",
            "(Tensor x, Scalar w) -> Tensor",
            "(Tensor x, float w) -> Tensor",
            "(Tensor x, float? w) -> Tensor",
            "(Tensor x, bool[] w) -> Tensor",
        ]

        def foo(x):
            ...

        def bar(x, w):
            ...

        for schema in schemas:
            op = custom_op(f"{TestCustomOp.test_ns}::foo", schema)(foo)
            op._destroy()
        for schema in other_schemas:
            op = custom_op(f"{TestCustomOp.test_ns}::bar", schema)(bar)
            op._destroy()

    def test_reserved_ns(self):
        from torch._custom_op.impl import RESERVED_NS

        for ns in RESERVED_NS:
            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):

                @custom_op(f"{ns}::foo", "(Tensor x) -> Tensor")
                def foo(x):
                    ...

            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):

                @custom_op(f"{ns}::foo2")
                def foo2(x: torch.Tensor) -> torch.Tensor:
                    ...

    def test_private_ctor(self):
        with self.assertRaisesRegex(RuntimeError, "CustomOp constructor is private"):
            CustomOp(None, None, None, None, None)

    def test_lifetime(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        # 3 references:
        # - foo (in this function)
        # - arg passed to sys.getrefcount
        # - global_registry
        self.assertEqual(sys.getrefcount(foo), 3)

        # We can't define an op multiple times,
        with self.assertRaisesRegex(RuntimeError, "multiple times"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: torch.Tensor) -> torch.Tensor:
                ...

        # Unless we delete the original op.
        foo._destroy()

        # Smoke test
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        foo._destroy()

    def test_autograd_notimplemented(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            foo(x)
        foo._destroy()

        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: Sequence[torch.Tensor]) -> torch.Tensor:
            ...

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            foo([y, x])
        foo._destroy()

        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            ...

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            foo(y, x)
        foo._destroy()

    def test_autograd_notimplemented_gradmode(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu"])
        def foo_impl(x, y):
            return x * y

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        with torch.no_grad():
            # Shouldn't raise, because we are in no_grad
            foo(y, x)

    def test_impl_cpu(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl("cpu")
        def foo_cpu(x):
            return x.sin()

        x = torch.randn(3)
        result = foo(x)
        self.assertEqual(result, foo_cpu(x))

    def test_impl_invalid_devices(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        def foo_impl(x):
            return x.sin()

        from torch._custom_op.impl import SUPPORTED_DEVICE_TYPE_TO_KEY

        for device_type in SUPPORTED_DEVICE_TYPE_TO_KEY.keys():
            # Smoke test: should not raise error
            foo.impl(device_type)(foo_impl)

        # Not supported by this API: we can either support them in the future
        # or provide some other CustomOp.def_* function. This depends on how
        # common the use cases are.
        for invalid_type in ["hip", "xla", "mkldnn", ["cpu", "hip"]]:
            with self.assertRaisesRegex(ValueError, "we only support device_type"):
                foo.impl(invalid_type)(foo_impl)
        foo._destroy()

    def test_backward_partially_registered(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sin()

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError, "unable to find a 'save_for_backward'"
        ):
            y = foo(x)
            y.backward()

    def test_save_for_backward_inputs_are_namedtuple(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sin()

        hit = 0

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            nonlocal hit
            hit += 1
            self.assertTrue(isinstance(inputs, tuple))
            self.assertEqual(list(inputs._asdict().keys()), ["x"])
            return inputs.x

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        y = foo(x)
        self.assertEqual(hit, 1)
        y.backward()
        self.assertEqual(hit, 1)

    def test_backward_returns_dict(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        y = foo(x)
        with self.assertRaisesRegex(RuntimeError, "to be a dict"):
            y.backward()

    def test_backward_dict_invalid_keys(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "y": None}

        x = torch.randn([], requires_grad=True)
        y = foo(x)
        with self.assertRaisesRegex(RuntimeError, "to have keys {'x'}"):
            y.backward()

    def test_backward_dict_grad_for_nontensor(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x, dim):
            return x.sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "dim": None}

        x = torch.randn([], requires_grad=True)
        y = foo(x, 32)
        with self.assertRaisesRegex(RuntimeError, "non-Tensor-like types"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_tensors(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x, y):
            return x.sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        y = foo(x, x)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_optional_tensors(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x, y):
            return x.sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        y = foo(x, None)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_grads_are_tensor_or_none(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"x": (grad * saved.cos(),)}

        x = torch.randn([], requires_grad=True)
        y = foo(x)
        with self.assertRaisesRegex(RuntimeError, "either None or a Tensor"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads_with_same_numel(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(xs):
            return xs[0].sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"xs": [grad * saved.cos(), None]}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        y = foo(xs)
        with self.assertRaisesRegex(RuntimeError, "3 gradients but got 2"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads_none_or_Tensor(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(xs):
            return xs[0].sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"xs": [grad * saved.cos(), None, (None,)]}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        y = foo(xs)
        with self.assertRaisesRegex(RuntimeError, "None or Tensor"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(xs):
            return xs[0].sin()

        @foo.impl_save_for_backward()
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]

        @foo.impl_backward()
        def foo_backward(ctx, saved, grad):
            return {"xs": None}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        y = foo(xs)
        with self.assertRaisesRegex(RuntimeError, "list of gradients"):
            y.backward()

    def test_backward_output_differentiability_type(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            ...

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @foo.impl_backward(output_differentiability=True)
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    def test_backward_output_differentiability_numel(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            ...

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @foo.impl_backward(output_differentiability=[True])
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_impl_separate(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl("cpu")
        def foo_cpu(x):
            return x.sin()

        @foo.impl("cuda")
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
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
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
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x, dim):
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        x = torch.randn(2, 3, device="meta")
        result = foo(x, 1)
        self.assertEqual(result.shape, foo_meta(x, 1).shape)
        foo._destroy()

    def test_duplicate_impl(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x, dim):
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        with self.assertRaisesRegex(
            RuntimeError, r"already has a abstract impl.*at .*test_custom_ops.py:\d+"
        ):

            @foo.impl_abstract()
            def foo_meta2(x, dim):
                output_shape = list(x.shape)
                del output_shape[dim]
                return x.new_empty(output_shape)

        foo._destroy()

    def test_new_data_dependent_symint(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x):
            ctx = torch._custom_op.impl.get_ctx()
            with self.assertRaisesRegex(ValueError, "greater than or equal to 2"):
                ctx.create_unbacked_symint(min=1)
            with self.assertRaisesRegex(ValueError, "greater than or equal to 2"):
                ctx.create_unbacked_symint(min=-1)
            with self.assertRaisesRegex(ValueError, "SymInt"):
                ctx.create_unbacked_symint(max=x.numel())
            return torch.clone(x)

        x = torch.randn(2, 3, device="cpu")
        make_fx(foo, tracing_mode="symbolic")(x)
        foo._destroy()

    def test_meta_for_data_dependent_shape_operation(self):
        from torch.testing._internal.custom_op_db import numpy_nonzero

        x = torch.randn(10, device="meta")
        with self.assertRaisesRegex(RuntimeError, "data-dependent output shape"):
            numpy_nonzero(x)

    def test_basic_make_fx(self):
        # More serious tests are in our CustomOp opinfo db,
        # this one is just a sanity check.
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl_abstract()
        def foo_meta(x):
            return x.sum()

        x = torch.randn(3)
        gm = make_fx(foo, tracing_mode="symbolic")(x)
        self.assertTrue(f"{TestCustomOp.test_ns}.foo" in gm.code)
        foo._destroy()

    def test_not_implemented_error(self):
        @custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        x = torch.randn(3)
        with self.assertRaisesRegex(NotImplementedError, "cpu impl registered"):
            foo(x)

        x = torch.randn(3, device="meta")
        with self.assertRaisesRegex(NotImplementedError, "abstract impl registered"):
            foo(x)

        @custom_op(f"{TestCustomOp.test_ns}::bar")
        def bar(sizes: Sequence[int]) -> torch.Tensor:
            ...

        with self.assertRaisesRegex(NotImplementedError, "no Tensor inputs"):
            bar((1, 2, 3))

    def test_abstract_registration_location(self):
        loc = torch.testing._internal.custom_op_db.numpy_nonzero._get_impl(
            "abstract"
        ).location
        matches = re.match(r".*custom_op_db.py:\d+", loc)
        self.assertIsNotNone(matches)

    def test_data_dependent_basic(self):
        from torch.testing._internal.custom_op_db import numpy_nonzero

        def f(x):
            return numpy_nonzero(x)

        x = torch.randn(5, 5)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        self.assertTrue("nonzero" in gm.code)

    def test_data_dependent_fake_tracing(self):
        from torch.testing._internal.custom_op_db import numpy_nonzero

        def f(x):
            return numpy_nonzero(x)

        x = torch.randn(5, 5)
        with self.assertRaises(
            torch._subclasses.fake_tensor.DynamicOutputShapeException
        ):
            make_fx(f, tracing_mode="fake")(x)

    def test_symints(self):
        def f(x):
            return torch.testing._internal.custom_op_db.numpy_view_copy(x, x.shape)

        x = torch.randn(2, 3, 4)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        result = gm(x)
        self.assertEqual(result, f(x))
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sym_size = torch.ops.aten.sym_size(x_1, 0)
    sym_size_1 = torch.ops.aten.sym_size(x_1, 1)
    sym_size_2 = torch.ops.aten.sym_size(x_1, 2)
    numpy_view_copy = torch.ops._torch_testing.numpy_view_copy.default(x_1, [sym_size, sym_size_1, sym_size_2]);  x_1 = sym_size = sym_size_1 = sym_size_2 = None
    return numpy_view_copy""",  # noqa: B950
        )

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
            dict(counters["graph_break"]),
            {"dynamic shape operator: _torch_testing.numpy_nonzero.default": 1},
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

        self.assertEqual(len(counters["graph_break"]), 0)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestCustomOpTesting, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
