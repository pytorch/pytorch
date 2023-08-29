# Owner(s): ["module: custom-operators"]

from torch.testing._internal.common_utils import *  # noqa: F403
from torch.testing._internal.common_device_type import *  # noqa: F403
import collections

import itertools
import re
import typing

import torch._custom_ops as custom_ops

import torch.testing._internal.custom_op_db
import torch.testing._internal.optests as optests
from functorch import make_fx
from torch import Tensor
from torch._custom_op.impl import custom_op, CustomOp
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.optests.compile_check import operator_compile_check
from typing import *  # noqa: F403


class CustomOpTestCaseBase(TestCase):
    test_ns = "_test_custom_op"

    def setUp(self):
        self.libraries = []

    def tearDown(self):
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{self.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            del lib.m
        del self.libraries

    def ns(self):
        return getattr(torch.ops, self.test_ns)

    def lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")
        self.libraries.append(result)
        return result

    def get_op(self, qualname):
        return torch._custom_op.impl.get_op(qualname)


@unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
class TestCustomOpTesting(CustomOpTestCaseBase):
    @parametrize("check_gradients", (False, "auto"))
    @parametrize("dynamic", (True, False))
    def test_aot_autograd_check_degenerate_cases(
        self, device, dynamic, check_gradients
    ):
        def simple(x):
            return x.clone()

        # Should not raise
        x = torch.randn(3, device=device)
        optests.aot_autograd_check(
            simple, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        def outputs_dont_require_grad(x):
            return x.detach()

        # Should not raise
        y = torch.randn(3, device=device, requires_grad=True)
        optests.aot_autograd_check(
            simple, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        def no_outputs(x):
            return x.detach()

        # Should not raise
        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn(3, device=device, requires_grad=False)
        optests.aot_autograd_check(
            no_outputs, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )
        optests.aot_autograd_check(
            no_outputs, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

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
            operator_compile_check(
                lambda x: self.get_op(f"{self.test_ns}::foo")(x), (x,), {}
            )

    def test_autograd_registration_check_autograd_kernel(self, device):
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
                return gx

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")

        x = torch.randn(3, requires_grad=True, device=device)
        # Should not raise
        optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_compositeimplicitautograd(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        def foo_impl(x):
            return x.sin().cos()

        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        # Should not raise
        optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_incorrect_composite(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        def foo_impl(x):
            return x.sin().cos()

        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_incorrect(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return torch.sin(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        lib.impl("foo", Foo.apply, "CPU")
        lib.impl("foo", Foo.apply, "CUDA")

        x = torch.randn(3, requires_grad=True, device=device)
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})

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


class TestCustomOp(CustomOpTestCaseBase):
    test_ns = "_test_custom_op"

    def test_invalid_schemas(self):
        # function schmea validation goes through torchgen, so this is just a
        # basic test.
        with self.assertRaisesRegex(AssertionError, "Invalid function schema: foo"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(")

    def test_invalid_qualname(self):
        with self.assertRaisesRegex(ValueError, "overload"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo.Tensor", "() -> ()")

    def test_name_must_match(self):
        with self.assertRaisesRegex(ValueError, "to have name"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def baz(x: Tensor) -> Tensor:
                raise NotImplementedError()

    def test_unsupported_schemas(self):
        with self.assertRaisesRegex(ValueError, "does not support non-functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a!) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "does not support view functions"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "no outputs"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor x) -> ()")(
                foo
            )
        with self.assertRaisesRegex(ValueError, "self"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor self) -> ()")(
                foo
            )

    # Tests for the older custom_op API
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

    # Tests for the older custom_op API
    def test_unsupported_annotation_categories(self):
        with self.assertRaisesRegex(ValueError, "varargs"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(*args):
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "varkwargs"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(**kwargs):
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "must have a type annotation"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x):
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "default value"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Optional[Tensor] = None):
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "default value"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Optional[Tensor] = None):
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported"):

            @custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor) -> Tuple[Tensor, ...]:
                raise NotImplementedError()

            del foo

    def _generate_examples(self, typ):
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
            assert len(args) == 2 and (args[0] is type(None) or args[1] is type(None))
            elt = args[0] if args[1] is type(None) else args[1]
            return self._generate_examples(elt) + [None]
        if origin is list:
            args = typing.get_args(typ)
            assert len(args) == 1
            elt = args[0]
            return [
                self._generate_examples(elt),
                self._generate_examples(elt),
                self._generate_examples(elt),
            ]
        if origin is collections.abc.Sequence:
            args = typing.get_args(typ)
            assert len(args) == 1
            examples = self._generate_examples(args[0])
            return list(itertools.product(examples, examples)) + []
        raise NotImplementedError(
            f"testrunner cannot generate instanstance of type {typ}"
        )

    def test_supported_return_types_single_return(self):
        for typ in torch._custom_op.impl.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> typ:
                        raise NotImplementedError()

                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> typ:
                        return example

                    op = self.get_op(f"{self.test_ns}::foo")
                    result = op(torch.randn([]))
                    self.assertEqual(result, example, msg=f"{typ} {example}")
                finally:
                    custom_ops._destroy(f"{self.test_ns}::foo")

    def test_supported_return_types_multi_return(self):
        for typ in torch._custom_op.impl.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> Tuple[typ, typ]:
                        raise NotImplementedError()

                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> Tuple[typ, typ]:
                        return (example, example)

                    op = self.get_op(f"{self.test_ns}::foo")
                    result = op(torch.randn([]))
                    expected = (example, example)
                    self.assertEqual(result, expected, msg=f"{typ} {example}")
                finally:
                    custom_ops._destroy(f"{self.test_ns}::foo")

    def test_supported_param_types(self):
        for typ in torch._custom_op.impl.SUPPORTED_PARAM_TYPES:

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: typ) -> Tensor:
                raise NotImplementedError()

            yeet = None

            @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=["cpu"])
            def foo_cpu(x, y):
                nonlocal yeet
                yeet = y
                return x.clone()

            try:
                for example in self._generate_examples(typ):
                    op = self.get_op(f"{self.test_ns}::foo")
                    op(torch.randn([]), example)
                    self.assertEqual(yeet, example, msg=f"{typ} {example}")
                    yeet = None
            finally:
                custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

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

        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
            raise NotImplementedError()

        called = 0

        @custom_ops.impl(f"{self.test_ns}::foo", device_types="cpu")
        def foo_cpu(x, sizes):
            nonlocal called
            called += 1
            # Dispatcher will normalize the sequence type into a List
            self.assertEqual(sizes, [1, 2, 3])
            return x.clone()

        x = torch.randn([])
        seq = MySequence()
        op = self.get_op(f"{self.test_ns}::foo")
        op(x, seq)
        self.assertEqual(called, 1)

    def test_unsupported_param_types(self):
        # Not comprehensive (it doesn't need to be), just a check that our mechanism works
        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: List[Optional[int]]) -> Tensor:
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # int[N] in Dispatcher is a bit wild, so we don't try to support it.
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # We could theoretically support this, but the syntax for suporting
            # int[] is Sequence[int]
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: List[int]) -> Tensor:
                raise NotImplementedError()

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Callable) -> Tensor:
                raise NotImplementedError()

            del foo

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

        for schema in schemas:
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", schema)
            custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        for schema in other_schemas:
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar", schema)
            custom_ops._destroy(f"{TestCustomOp.test_ns}::bar")

    def test_reserved_ns(self):
        from torch._custom_op.impl import RESERVED_NS

        for ns in RESERVED_NS:
            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):
                custom_ops.custom_op(f"{ns}::foo", "(Tensor x) -> Tensor")

            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):

                @custom_ops.custom_op(f"{ns}::foo2")
                def foo2(x: torch.Tensor) -> torch.Tensor:
                    raise NotImplementedError()

    def test_private_ctor(self):
        with self.assertRaisesRegex(RuntimeError, "CustomOp constructor is private"):
            CustomOp(None, None, None, None, None)

    def test_lifetime(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        custom_op = torch._custom_op.impl.get_op(f"{TestCustomOp.test_ns}::foo")

        # We can't define an op multiple times,
        with self.assertRaisesRegex(RuntimeError, "multiple times"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
                raise NotImplementedError()

        # Unless we delete the original op.
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

        # Smoke test
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            raise NotImplementedError()

        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            raise NotImplementedError()

        x = torch.randn(3, requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError()

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op([y, x])
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(y, x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented_gradmode(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x * y

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with torch.no_grad():
            # Shouldn't raise, because we are in no_grad
            op(y, x)

    def test_impl_cpu(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cpu")
        def foo_cpu(x):
            return x.sin()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x)
        self.assertEqual(result, foo_cpu(x))

    def test_impl_invalid_devices(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        def foo_impl(x):
            return x.sin()

        from torch._custom_op.impl import SUPPORTED_DEVICE_TYPE_TO_KEY

        for device_type in SUPPORTED_DEVICE_TYPE_TO_KEY.keys():
            # Smoke test: should not raise error
            custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=device_type)(
                foo_impl
            )

        # Not supported by this API: we can either support them in the future
        # or provide some other CustomOp.def_* function. This depends on how
        # common the use cases are.
        for invalid_type in ["hip", "xla", "mkldnn", ["cpu", "hip"]]:
            with self.assertRaisesRegex(ValueError, "we only support device_type"):
                custom_ops.impl(
                    f"{TestCustomOp.test_ns}::foo", device_types=invalid_type
                )(foo_impl)

    def test_backward_partially_registered(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(
            RuntimeError, "unable to find a 'save_for_backward'"
        ):
            y = op(x)
            y.backward()

    def test_save_for_backward_inputs_are_namedtuple(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        hit = 0

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            nonlocal hit
            hit += 1
            self.assertTrue(isinstance(inputs, tuple))
            self.assertEqual(list(inputs._asdict().keys()), ["x"])
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        self.assertEqual(hit, 1)
        y.backward()
        self.assertEqual(hit, 1)

    def test_backward_returns_dict(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "to be a dict"):
            y.backward()

    def test_backward_dict_invalid_keys(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "y": None}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "to have keys {'x'}"):
            y.backward()

    def test_backward_dict_grad_for_nontensor(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, dim):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "dim": None}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, 32)
        with self.assertRaisesRegex(RuntimeError, "non-Tensor-like types"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_tensors(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, x)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_optional_tensors(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, None)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_grads_are_tensor_or_none(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": (grad * saved.cos(),)}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "either None or a Tensor"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads_with_same_numel(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            return xs[0].sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"xs": [grad * saved.cos(), None]}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(xs)
        with self.assertRaisesRegex(RuntimeError, "3 gradients but got 2"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads_none_or_Tensor(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            return xs[0].sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"xs": [grad * saved.cos(), None, (None,)]}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(xs)
        with self.assertRaisesRegex(RuntimeError, "None or Tensor"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            return xs[0].sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"xs": None}

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(xs)
        with self.assertRaisesRegex(RuntimeError, "list of gradients"):
            y.backward()

    def test_backward_output_differentiability_type(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError()

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=True
            )
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    def test_backward_output_differentiability_numel(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            raise NotImplementedError()

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=[True]
            )
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    def test_backward_output_differentiability_tensorlist(self):
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            raise NotImplementedError()

        @custom_ops.impl(f"{self.test_ns}::foo")
        def foo_impl(x):
            return [x.clone(), x.clone()], x.clone()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return []

        @custom_ops.impl_backward(
            f"{TestCustomOp.test_ns}::foo", output_differentiability=[False, True]
        )
        def foo_backward(ctx, saved, grad_lst, grad):
            return {"x": grad}

        op = self.get_op(f"{self.test_ns}::foo")
        x = torch.randn(3, requires_grad=True)
        [a, b], c = op(x)
        self.assertFalse(a.requires_grad)
        self.assertFalse(b.requires_grad)
        self.assertTrue(c.requires_grad)

    def test_backward_output_differentiability_non_tensor(self):
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: Tensor) -> Tuple[Tensor, int]:
            raise NotImplementedError()

        @custom_ops.impl(f"{self.test_ns}::foo")
        def foo_impl(x):
            return x.clone(), 3

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return []

        @custom_ops.impl_backward(
            f"{TestCustomOp.test_ns}::foo", output_differentiability=[True, True]
        )
        def foo_backward(ctx, saved, grad0, grad1):
            return {"x": grad0}

        op = self.get_op(f"{self.test_ns}::foo")
        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "is not a Tensor"):
            op(x)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_impl_separate(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cpu")
        def foo_cpu(x):
            return x.sin()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cuda")
        def foo_cuda(x):
            return x.cos()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x)
        self.assertEqual(result, foo_cpu(x))

        x_cuda = x.cuda()
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x_cuda)
        self.assertEqual(result, foo_cuda(x_cuda))

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_impl_multiple(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.cos()

        op = self.get_op(f"{self.test_ns}::foo")
        x = torch.randn(3)
        result = op(x)
        self.assertEqual(result, foo_impl(x))

        x_cuda = x.cuda()
        result = op(x_cuda)
        self.assertEqual(result, foo_impl(x_cuda))

    def test_impl_meta(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl_abstract(f"{TestCustomOp.test_ns}::foo")
        def foo_meta(x, dim):
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        x = torch.randn(2, 3, device="meta")
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x, 1)
        self.assertEqual(result.shape, foo_meta(x, 1).shape)

    def test_duplicate_impl(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl_abstract(f"{TestCustomOp.test_ns}::foo")
        def foo_meta(x, dim):
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        with self.assertRaisesRegex(
            RuntimeError, r"already has a abstract impl.*at .*test_custom_ops.py:\d+"
        ):

            @custom_ops.impl_abstract(f"{TestCustomOp.test_ns}::foo")
            def foo_meta2(x, dim):
                output_shape = list(x.shape)
                del output_shape[dim]
                return x.new_empty(output_shape)

    def test_new_data_dependent_symint(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl_abstract(f"{TestCustomOp.test_ns}::foo")
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
        op = self.get_op(f"{self.test_ns}::foo")
        make_fx(op, tracing_mode="symbolic")(x)

    def test_meta_for_data_dependent_shape_operation(self):
        x = torch.randn(10, device="meta")
        with self.assertRaisesRegex(RuntimeError, "data-dependent output shape"):
            torch.ops._torch_testing.numpy_nonzero(x)

    def test_basic_make_fx(self):
        # More serious tests are in our CustomOp opinfo db,
        # this one is just a sanity check.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        @custom_ops.impl_abstract(f"{TestCustomOp.test_ns}::foo")
        def foo_meta(x):
            return x.sum()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        gm = make_fx(op, tracing_mode="symbolic")(x)
        self.assertTrue(f"{TestCustomOp.test_ns}.foo" in gm.code)

    def test_not_implemented_error(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(NotImplementedError, "cpu impl registered"):
            op(x)

        x = torch.randn(3, device="meta")
        with self.assertRaisesRegex(NotImplementedError, "abstract impl registered"):
            op(x)

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar")
        def bar(sizes: Sequence[int]) -> torch.Tensor:
            raise NotImplementedError()

        op = self.get_op(f"{self.test_ns}::bar")
        with self.assertRaisesRegex(NotImplementedError, "no Tensor inputs"):
            op((1, 2, 3))

    def test_abstract_registration_location(self):
        custom_op = torch._custom_op.impl._find_custom_op(
            "_torch_testing::numpy_nonzero"
        )
        loc = custom_op._get_impl("abstract").location
        matches = re.match(r".*custom_op_db.py:\d+", loc)
        self.assertIsNotNone(matches)

    def test_data_dependent_basic(self):
        def f(x):
            return torch.ops._torch_testing.numpy_nonzero(x)

        x = torch.randn(5, 5)
        gm = make_fx(f, tracing_mode="symbolic")(x)
        self.assertTrue("nonzero" in gm.code)

    def test_data_dependent_fake_tracing(self):
        def f(x):
            return torch.ops._torch_testing.numpy_nonzero(x)

        x = torch.randn(5, 5)
        with self.assertRaises(
            torch._subclasses.fake_tensor.DynamicOutputShapeException
        ):
            make_fx(f, tracing_mode="fake")(x)

    def test_symints(self):
        def f(x):
            return torch.ops._torch_testing.numpy_view_copy(x, x.shape)

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

    def test_impl_on_existing_op(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        @torch._custom_ops.impl(qualname)
        def foo_impl(x):
            return x.sin()

        op = self.get_op(qualname)
        x = torch.randn(3)
        result = op(x)
        self.assertEqual(result, x.sin())

    @parametrize(
        "key", ["CPU", "CUDA", "CompositeImplicitAutograd", "CompositeExplicitAutograd"]
    )
    def test_impl_on_existing_op_with_cpu_registration(self, key):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", foo_impl, key)
        op = self.get_op(qualname)

        with self.assertRaisesRegex(RuntimeError, "already has an implementation"):
            custom_ops.impl(qualname, func=foo_impl)

    def test_abstract_impl_on_existing_op(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        @torch._custom_ops.impl_abstract(qualname)
        def foo_impl(x):
            return x.sin()

        op = self.get_op(qualname)
        with torch._subclasses.FakeTensorMode():
            x = torch.randn(3)
            result = op(x)
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(result.stride(), x.stride())

    def test_abstract_impl_on_existing_op_with_meta(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", foo_impl, "Meta")
        op = self.get_op(qualname)

        with self.assertRaisesRegex(RuntimeError, r"already has .*Meta implementation"):
            custom_ops.impl_abstract(qualname, func=foo_impl)

    def test_abstract_impl_on_existing_op_with_CompositeImplicitAutograd(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")
        op = self.get_op(qualname)

        with self.assertRaisesRegex(RuntimeError, "CompositeImplicitAutograd"):
            custom_ops.impl_abstract(qualname, func=foo_impl)

    def test_abstract_impl_on_existing_op_with_CompositeExplicitAutograd(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")
        op = self.get_op(qualname)

        custom_ops.impl_abstract(qualname, func=lambda x: x.sum())
        with torch._subclasses.FakeTensorMode():
            x = torch.randn(10)
            result = op(x)
            self.assertEqual(result.shape, ())

    def _test_backward_impl_raises(self, qualname, err_regex):
        with self.assertRaisesRegex(RuntimeError, err_regex):

            @custom_ops.impl_save_for_backward(qualname)
            def foo2(x):
                return

        with self.assertRaisesRegex(RuntimeError, err_regex):

            @custom_ops.impl_backward(qualname)
            def foo3(x):
                return

    def test_backward_impl_on_existing_op_incorrect_schema_views(self):
        lib = self.lib()
        lib.define("foo(Tensor(a) x) -> Tensor(a)")
        qualname = f"{self.test_ns}::foo"
        self._test_backward_impl_raises(qualname, "operator that returns views")

    def test_backward_impl_on_existing_op_incorrect_schema_mutable(self):
        lib = self.lib()
        lib.define("foo(Tensor(a!) x) -> Tensor")
        qualname = f"{self.test_ns}::foo"
        self._test_backward_impl_raises(qualname, "non-functional")

    def test_backward_impl_on_existing_op_incorrect_schema_no_output(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> ()")
        qualname = f"{self.test_ns}::foo"
        self._test_backward_impl_raises(qualname, "no returns")

    def test_backward_impl_on_existing_op_CompositeImplicitAutograd(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"
        lib.impl("foo", lambda x: x.sin().cos(), "CompositeImplicitAutograd")
        self._test_backward_impl_raises(qualname, "CompositeImplicitAutograd")

    @parametrize("key", ["Autograd", "AutogradCPU", "AutogradCUDA"])
    def test_backward_impl_on_existing_op_with_key(self, key):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"
        lib.impl("foo", lambda x: x.sin().cos(), key)
        self._test_backward_impl_raises(qualname, key)

    def test_backward_impl_on_existing_op(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        @custom_ops.impl(qualname)
        def foo_impl(x):
            with torch.no_grad():
                return x.sin()

        @custom_ops.impl_save_for_backward(qualname)
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(qualname)
        def foo_backward(ctx, saved, grad_out):
            return {"x": grad_out * saved.cos()}

        op = self.get_op(qualname)
        x = torch.randn([], requires_grad=True)
        y = op(x)
        (gx,) = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())


def op_with_incorrect_schema(testcase, name):
    lib = testcase.lib()
    lib.define(f"{name}(Tensor x) -> Tensor")
    qualname = f"{testcase.test_ns}::{name}"
    lib.impl(name, lambda x: x[:], "CompositeExplicitAutograd")
    return testcase.get_op(qualname)


class MiniOpTest(CustomOpTestCaseBase):
    test_ns = "mini_op_test"

    def _op_delayed_backward_error(self, name):
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::{name}"
        lib.impl(name, lambda x: x.clone(), "CompositeExplicitAutograd")
        op = self.get_op(qualname)

        class Op(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                import ctypes

                c_float_p = ctypes.POINTER(ctypes.c_float)
                contents = ctypes.cast(x.data_ptr(), c_float_p).contents
                print(contents)
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, grad):
                raise NotImplementedError()

        def autograd_impl(x):
            return Op.apply(x)

        lib.impl(name, autograd_impl, "Autograd")
        return op

    def _op_with_no_abstract_impl(self, name):
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::{name}"
        lib.impl(name, lambda x: x.clone(), "CPU")
        return self.get_op(qualname)

    def test_mm(self):
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(3, 5)
        result = torch.ops.aten.mm.default(x, y)
        self.assertEqual(result, x @ y)

    def test_mm_errors(self):
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(4, 5)
        with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
            result = torch.ops.aten.mm.default(x, y)

    def test_nonzero(self):
        x = torch.tensor([0, 1, 2, 0, 0])
        y = torch.ops.aten.nonzero.default(x)
        self.assertEqual(y, torch.tensor([[1], [2]]))

    def test_inplace(self):
        x = torch.randn(3)
        x_clone = x.clone()
        y = torch.ops.aten.sin_(x)
        self.assertEqual(x, x_clone.sin())

    def test_incorrect_schema(self):
        op = op_with_incorrect_schema(self, "incorrect_schema")
        x = torch.randn(3)
        op(x)

    def test_no_abstract(self):
        op = self._op_with_no_abstract_impl("no_abstract")
        x = torch.randn(3)
        op(x)

    def test_delayed_error(self):
        op = self._op_delayed_backward_error("delayed_error")
        x = torch.randn([], requires_grad=True)
        y = op(x)
        with self.assertRaises(NotImplementedError):
            y.sum().backward()

    def test_delayed_error_no_requires_grad(self):
        op = self._op_delayed_backward_error("delayed_error")
        x = torch.randn([])
        y = op(x)


mini_op_test_failures_dict = {
    "aten::nonzero": {
        # Nonzero doesn't support static shapes
        "test_aot_dispatch_static__test_nonzero": "xfail",
    },
    "mini_op_test::delayed_error": {
        "test_aot_dispatch_dynamic__test_delayed_error": "xfail",
        "test_aot_dispatch_static__test_delayed_error": "xfail",
    },
    "mini_op_test::incorrect_schema": {
        # "skip" just to test the skip mechanism
        "test_schema__test_incorrect_schema": "skip",
    },
    "mini_op_test::no_abstract": {
        "test_aot_dispatch_dynamic__test_no_abstract": "xfail",
        "test_aot_dispatch_static__test_no_abstract": "xfail",
        "test_faketensor__test_no_abstract": "xfail",
    },
}

mini_op_test_checks = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_static",
    "test_aot_dispatch_dynamic",
]

optests.generate_opcheck_tests(
    MiniOpTest,
    ["aten", "MiniOpTest"],
    mini_op_test_failures_dict,
    "test/test_custom_ops.py",
    [],
    mini_op_test_checks,
)


class TestGenerateOpcheckTests(CustomOpTestCaseBase):
    def test_MiniOpTest(self):
        for orig_test in ["test_mm", "test_nonzero"]:
            for test in mini_op_test_checks:
                expected_test = f"{test}__{orig_test}"
                self.assertTrue(hasattr(MiniOpTest, expected_test), msg=expected_test)

    def test_failures_dict_validation(self):
        from torch.testing._internal.optests.generate_tests import (
            validate_failures_dict,
        )

        failures = {
            "mini_op_test::incorrect_schema": {},
            "mini_op_test::delayed_error": {},
        }
        with self.assertRaisesRegex(RuntimeError, "alphabetical"):
            validate_failures_dict(failures, mini_op_test_checks, MiniOpTest)

        failures = {
            "mini_op_test::incorrect_schema": {
                "test_aot_dispatch_static__test_delayed_error": "xfail",
                "test_aot_dispatch_dynamic__test_delayed_error": "xfail",
            }
        }
        with self.assertRaisesRegex(RuntimeError, "alphabetical"):
            validate_failures_dict(failures, mini_op_test_checks, MiniOpTest)

        failures = {
            "mini_op_test::incorrect_schema": {
                "test_aot_dispatch_static__test_delayed_error": "XFAIL",
            }
        }
        with self.assertRaisesRegex(RuntimeError, "got value=XFAIL"):
            validate_failures_dict(failures, mini_op_test_checks, MiniOpTest)

        failures = {
            "mini_op_test::incorrect_schema": {
                "test_aot_dispatch__test_delayed_error": "xfail",
            }
        }
        with self.assertRaisesRegex(RuntimeError, "should begin with one of"):
            validate_failures_dict(failures, mini_op_test_checks, MiniOpTest)

        failures = {
            "mini_op_test::incorrect_schema": {
                "test_aot_dispatch_static__test_delayed_error_nopenopenope": "xfail",
            }
        }
        with self.assertRaisesRegex(RuntimeError, "does not exist on the TestCase"):
            validate_failures_dict(failures, mini_op_test_checks, MiniOpTest)

    def test_opcheck(self):
        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "OpOverload"):
            optests.opcheck(torch.sin, (x,))
        with self.assertRaisesRegex(ValueError, "test_utils to be subset of"):
            optests.opcheck(torch.ops.aten.sin.default, (x,), test_utils="blah")
        result = optests.opcheck(torch.ops.aten.sin.default, (x,))

        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_static": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )

        result = optests.opcheck(
            torch.ops.aten.sin.default, (x,), test_utils="test_schema"
        )
        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
            },
        )

        result = optests.opcheck(
            torch.ops.aten.sin.default,
            (x,),
            test_utils=["test_schema", "test_faketensor"],
        )
        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_faketensor": "SUCCESS",
            },
        )

    def test_opcheck_bad_op(self):
        op = op_with_incorrect_schema(self, "foo")
        x = torch.randn(3)
        with self.assertRaisesRegex(Exception, "is not defined to alias output"):
            optests.opcheck(op, (x,))

        result = optests.opcheck(op, (x,), raise_exception=False)
        self.assertTrue(isinstance(result["test_schema"], RuntimeError))
        del result["test_schema"]
        self.assertEqual(
            result,
            {
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_static": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestCustomOpTesting, globals(), only_for=only_for)
instantiate_parametrized_tests(TestCustomOp)

if __name__ == "__main__":
    run_tests()
