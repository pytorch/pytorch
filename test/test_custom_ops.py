# Owner(s): ["module: custom-operators"]

from torch.testing._internal.common_utils import *  # noqa: F403
from torch.testing._internal.common_device_type import *  # noqa: F403
import collections

import itertools
import os
import re
import typing

import torch._custom_ops as custom_ops

import torch.testing._internal.optests as optests
import torch.utils.cpp_extension

from functorch import make_fx
from torch import Tensor
from torch._custom_op.impl import custom_op, CustomOp, infer_schema
from torch._library.infer_schema import tuple_to_list
from torch._utils_internal import get_file_path_2
from torch.testing._internal import custom_op_db
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.custom_op_db import numpy_nonzero
from typing import *  # noqa: F403
import numpy as np


def requires_compile(fun):
    fun = unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")(fun)
    return fun


class CustomOpTestCaseBase(TestCase):
    test_ns = "_test_custom_op"

    def setUp(self):
        super().setUp()
        self.libraries = []

    def tearDown(self):
        super().tearDown()
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{self.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            lib._destroy()
        del self.libraries

    def ns(self):
        return getattr(torch.ops, self.test_ns)

    def lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        self.libraries.append(result)
        return result

    def get_op(self, qualname):
        return torch._custom_op.impl.get_op(qualname)


@requires_compile
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

        x = torch.tensor(3.14159 / 3, requires_grad=True, device=device)
        with self.assertRaisesRegex(
            optests.OpCheckError, "Argument x is not defined as mutable but was mutated"
        ):
            torch.library.opcheck(op, (x,), {})

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

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "Argument x is not defined to alias output but was aliasing",
        ):
            torch.library.opcheck(op, (x,), {})

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

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "_test_custom_op.foo.default",
        ):
            torch.library.opcheck(op, (x,), {})

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
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

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(optests.OpCheckError, "Shapes .* are not equal"):
            torch.library.opcheck(op, (x,), {})

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

        x = torch.tensor([0, 1.0])
        y = x.clone()
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "We only support functionalizing operators whose outputs do not have alias annotations",
        ):
            torch.library.opcheck(op, (y,), {})

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

        x = torch.randn([], requires_grad=True)

        with self.assertRaisesRegex(
            torch.testing._internal.optests.OpCheckError,
            "does not have an autograd kernel",
        ):
            torch.library.opcheck(op, (x,), {})

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

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError, "eager-mode PyTorch vs AOTAutograd"
        ):
            torch.library.opcheck(op, (x,), {})

    @ops(custom_op_db.custom_op_db, dtypes=OpDTypes.any_one)
    def test_opcheck_opinfo(self, device, dtype, op):
        for sample_input in op.sample_inputs(
            device, dtype, requires_grad=op.supports_autograd
        ):
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            torch.library.opcheck(
                op.op,
                args,
                kwargs,
            )

    def test_opcheck_fails_basic(self, device):
        @custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sum()

        x = torch.randn(3, device=device, requires_grad=True)
        # Triggers the CustomOp autograd NYI error
        with self.assertRaisesRegex(
            optests.OpCheckError, "Autograd has not been implemented for operator"
        ):
            torch.library.opcheck(self.get_op(f"{self.test_ns}::foo"), (x,), {})

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

    @requires_compile
    def test_functionalize_error(self):
        with torch.library._scoped_library(self.test_ns, "FRAGMENT") as lib:
            lib.define("foo(Tensor(a!) x) -> Tensor(a!)")

            def foo(x):
                return x.sin_()

            lib.impl("foo", foo, "CompositeExplicitAutograd")
            foo_op = self.get_op(f"{self.test_ns}::foo")

            lib.define("bar(Tensor(a) x) -> Tensor(a)")

            def bar(x):
                return x.view(-1)

            lib.impl("bar", bar, "CompositeExplicitAutograd")
            bar_op = self.get_op(f"{self.test_ns}::bar")

            msg = r".*We only support functionalizing operators whose outputs do not have alias annotations"

            x = torch.randn(3)

            @torch.compile(backend="aot_eager", fullgraph=True)
            def f(x):
                return foo_op(x)

            @torch.compile(backend="aot_eager", fullgraph=True)
            def g(x):
                return bar_op(x)

            with self.assertRaisesRegex(RuntimeError, msg):
                f(x)
            with self.assertRaisesRegex(RuntimeError, msg):
                g(x)

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
                raise NotImplementedError

    def test_unsupported_schemas(self):
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a!) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
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

    def test_infer_schema_supported(self):
        def a(x: Tensor) -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(infer_schema(a), """(Tensor x) -> Tensor""")

        def kwonly1(x: Tensor, *, y: int, z: float) -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(
            infer_schema(kwonly1), """(Tensor x, *, SymInt y, float z) -> Tensor"""
        )

        def kwonly2(*, y: Tensor) -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(infer_schema(kwonly2), """(*, Tensor y) -> Tensor""")

        def b(
            x: Tensor,
            y: int,
            z: bool,
            a: float,
            b: torch.dtype,
            c: torch.device,
            d: torch.types.Number,
        ) -> Tuple[Tensor, int, float, bool]:
            return torch.empty([]), 1, 0.1, True

        self.assertExpectedInline(
            infer_schema(b),
            """(Tensor x, SymInt y, bool z, float a, ScalarType b, Device c, Scalar d) -> (Tensor, SymInt, float, bool)""",
        )

        def c(
            x: Tensor,
            y: Sequence[Tensor],
            z: Optional[Tensor],
            w: Sequence[Optional[Tensor]],
        ) -> List[Tensor]:
            return [torch.empty([])]

        self.assertExpectedInline(
            infer_schema(c),
            """(Tensor x, Tensor[] y, Tensor? z, Tensor?[] w) -> Tensor[]""",
        )

        def d(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            return [torch.empty([])], torch.empty([])

        self.assertExpectedInline(
            infer_schema(d), """(Tensor x) -> (Tensor[], Tensor)"""
        )

        def e() -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(infer_schema(e), """() -> Tensor""")

        def f(x: Tensor) -> None:
            pass

        self.assertExpectedInline(infer_schema(f), """(Tensor x) -> ()""")

        def g(
            x: Tensor, y: List[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            pass

        self.assertExpectedInline(
            infer_schema(g), """(Tensor x, Tensor[] y, Tensor[] z, Tensor?[] w) -> ()"""
        )

        self.assertExpectedInline(
            infer_schema(g, mutates_args={"x", "w", "z"}),
            """(Tensor(a0!) x, Tensor[] y, Tensor(a2!)[] z, Tensor(a3!)?[] w) -> ()""",
        )

    def test_infer_schema_unsupported(self):
        with self.assertRaisesRegex(ValueError, "varargs"):

            def foo(*args):
                raise NotImplementedError

            infer_schema(foo)

        with self.assertRaisesRegex(ValueError, "varkwargs"):

            def foo(**kwargs):
                raise NotImplementedError

            infer_schema(foo)

        with self.assertRaisesRegex(ValueError, "must have a type annotation"):

            def foo(x):
                raise NotImplementedError

            infer_schema(foo)

        with self.assertRaisesRegex(ValueError, "unsupported"):

            def foo(x: Tensor) -> Tuple[Tensor, ...]:
                raise NotImplementedError

            infer_schema(foo)

        with self.assertRaisesRegex(ValueError, "can be mutated"):

            def foo(x: Tensor, y: int) -> Tensor:
                raise NotImplementedError

            infer_schema(foo, mutates_args={"y"})

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
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> typ:
                        raise NotImplementedError

                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> typ:
                        return example

                    op = self.get_op(f"{self.test_ns}::foo")
                    result = op(torch.randn([]))
                    self.assertEqual(result, example, msg=f"{typ} {example}")
                finally:
                    custom_ops._destroy(f"{self.test_ns}::foo")

    def test_supported_return_types_multi_return(self):
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> Tuple[typ, typ]:
                        raise NotImplementedError

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
        for typ in torch._library.infer_schema.SUPPORTED_PARAM_TYPES:

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: typ) -> Tensor:
                raise NotImplementedError

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
            raise NotImplementedError

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
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # int[N] in Dispatcher is a bit wild, so we don't try to support it.
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, r"For example, typing.List\[int\]"):
            # test that we propose a correct and supported type.
            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaises(ValueError) as cm:

            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, float]) -> Tensor:
                raise NotImplementedError

            del foo

            self.assertNotIn("example", str(cm.exception), "")

        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Callable) -> Tensor:
                raise NotImplementedError

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
                    raise NotImplementedError

    def test_private_ctor(self):
        with self.assertRaisesRegex(RuntimeError, "CustomOp constructor is private"):
            CustomOp(None, None, None, None, None)

    def test_lifetime(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        custom_op = torch._custom_op.impl.get_op(f"{TestCustomOp.test_ns}::foo")

        # We can't define an op multiple times,
        with self.assertRaisesRegex(RuntimeError, "multiple times"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
                raise NotImplementedError

        # Unless we delete the original op.
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

        # Smoke test
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            raise NotImplementedError

        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op([y, x])
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(y, x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented_gradmode(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=True
            )
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    def test_backward_output_differentiability_numel(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            raise NotImplementedError

        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):

            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=[True]
            )
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    def test_backward_output_differentiability_tensorlist(self):
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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
            raise NotImplementedError

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

    def test_impl_abstract_overload(self):
        lib = self.lib()
        lib.define("sin.blah(Tensor x) -> Tensor")

        torch.library.impl_abstract(
            f"{self.test_ns}::sin.blah", torch.empty_like, lib=lib
        )

        op = self.ns().sin.blah
        x = torch.randn(3, device="meta")
        op(x)

    def test_impl_meta(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            raise NotImplementedError

        @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
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
            raise NotImplementedError

        @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x, dim):
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        with self.assertRaisesRegex(RuntimeError, r"test_custom_ops.py:\d+"):

            @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
            def foo_meta2(x, dim):
                output_shape = list(x.shape)
                del output_shape[dim]
                return x.new_empty(output_shape)

    def test_new_data_dependent_symint(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x):
            ctx = torch.library.get_ctx()
            r = ctx.new_dynamic_size(min=1)
            with self.assertRaisesRegex(ValueError, "greater than or equal to 0"):
                ctx.new_dynamic_size(min=-1)
            with self.assertRaisesRegex(ValueError, "SymInt"):
                ctx.new_dynamic_size(max=x.numel())
            # NB: You must return dynamic sizes!
            return x.new_empty(r)

        x = torch.randn(2, 3, device="cpu")
        op = self.get_op(f"{self.test_ns}::foo")
        make_fx(op, tracing_mode="symbolic")(x)

    def test_meta_for_data_dependent_shape_operation(self):
        x = torch.randn(10, device="meta")
        with self.assertRaisesRegex(RuntimeError, "data-dependent output shape"):
            numpy_nonzero(x)

    def test_basic_make_fx(self):
        # More serious tests are in our CustomOp opinfo db,
        # this one is just a sanity check.
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x):
            return x.sum()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        gm = make_fx(op, tracing_mode="symbolic")(x)
        self.assertTrue(f"{TestCustomOp.test_ns}.foo" in gm.code)

    def test_not_implemented_error(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(NotImplementedError, "cpu impl registered"):
            op(x)

        x = torch.randn(3, device="meta")
        with self.assertRaisesRegex(NotImplementedError, "no fake impl or Meta kernel"):
            op(x)

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar")
        def bar(sizes: Sequence[int]) -> torch.Tensor:
            raise NotImplementedError

        op = self.get_op(f"{self.test_ns}::bar")
        with self.assertRaisesRegex(NotImplementedError, "no Tensor inputs"):
            op((1, 2, 3))

    def test_data_dependent_basic(self):
        x = torch.randn(5, 5)
        gm = make_fx(numpy_nonzero, tracing_mode="symbolic")(x)
        self.assertTrue("nonzero" in gm.code)

    def test_data_dependent_fake_tracing(self):
        x = torch.randn(5, 5)
        # We've updated to attempt to use unbacked symints even for fake
        # tracing
        make_fx(numpy_nonzero, tracing_mode="fake")(x)

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
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(x_1, 2)
    numpy_view_copy = torch.ops._torch_testing.numpy_view_copy.default(x_1, [sym_size_int, sym_size_int_1, sym_size_int_2]);  x_1 = sym_size_int = sym_size_int_1 = sym_size_int_2 = None
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
            return numpy_nonzero(x.clone()).clone()

        f(torch.randn(10))

        self.assertEqual(len(counters["graph_break"]), 1)
        self.assertEqual(next(iter(counters["graph_break"].values())), 1)
        self.assertExpectedInline(
            next(iter(counters["graph_break"].keys())).replace(";", "\n"),
            """\
dynamic shape operator: _torch_testing.numpy_nonzero.default
 to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True""",
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

        @torch.library.impl_abstract(qualname, lib=self.lib())
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
            torch.library.impl_abstract(qualname, func=foo_impl, lib=self.lib())

    def test_abstract_impl_on_existing_op_with_CompositeImplicitAutograd(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")
        op = self.get_op(qualname)

        with self.assertRaisesRegex(RuntimeError, "CompositeImplicitAutograd"):
            torch.library.impl_abstract(qualname, func=foo_impl, lib=self.lib())

    def test_abstract_impl_on_existing_op_with_CompositeExplicitAutograd(self):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        qualname = f"{self.test_ns}::foo"

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")
        op = self.get_op(qualname)

        torch.library.impl_abstract(qualname, func=lambda x: x.sum(), lib=self.lib())
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

    def test_is_functional_schema(self):
        tests = {
            "foo(Tensor x) -> Tensor": True,
            "foo(Tensor(a) x) -> Tensor": True,
            "foo(Tensor(a!) x) -> Tensor": False,
            "foo(Tensor(a) x) -> Tensor(a)": False,
            "foo(Tensor x) -> ()": False,
        }
        for schema_str, expected in tests.items():
            res = torch._library.utils.is_functional_schema(schema_str)
            self.assertEqual(res, expected)

            from torchgen.model import FunctionSchema

            schema = FunctionSchema.parse(schema_str)
            res = torch._library.utils.is_functional_schema(schema)
            self.assertEqual(res, expected)

            schema = torch._C.parse_schema(schema_str)
            res = torch._library.utils.is_functional_schema(schema)
            self.assertEqual(res, expected)

    def test_incorrect_schema_types(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            with self.assertRaisesRegex(RuntimeError, "unknown type specifier"):
                lib.define("foo12(Tensor a) -> asdfasdf")
            with self.assertRaisesRegex(RuntimeError, "unknown type specifier"):
                lib.define("foo12(asdf a) -> Tensor")
            with self.assertRaisesRegex(RuntimeError, "Use `SymInt` or `int`"):
                lib.define("foo12(int64_t a) -> Tensor")
            with self.assertRaisesRegex(RuntimeError, "Use `float`"):
                lib.define("foo12(double a) -> Tensor")

    def test_is_tensorlist_like_type(self):
        tensorlists = [
            # Tensor[]
            torch.ops.aten.where.default._schema.returns[0].type,
            # Tensor?[]
            torch.ops.aten.index.Tensor._schema.arguments[1].type,
            # Tensor[]?
            torch._C.parse_schema("foo(Tensor[]? x) -> ()").arguments[0].type,
            # Tensor?[]?
            torch._C.parse_schema("foo(Tensor?[]? x) -> ()").arguments[0].type,
        ]
        non_tensorlists = [
            # Tensor
            torch.ops.aten.sin.default._schema.arguments[0].type,
            # IntList
            torch.ops.aten.sum.dim_IntList._schema.arguments[1].type,
        ]
        for a in tensorlists:
            self.assertTrue(torch._library.utils.is_tensorlist_like_type(a))
        for a in non_tensorlists:
            self.assertFalse(torch._library.utils.is_tensorlist_like_type(a))

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

    @parametrize(
        "tags",
        [
            subtest(torch.Tag.pointwise, "single"),
            subtest((torch.Tag.pointwise,), "tuple"),
            subtest([torch.Tag.pointwise], "list"),
        ],
    )
    def test_define_with_tags(self, tags):
        lib = self.lib()
        tags = (torch.Tag.pointwise,)
        torch.library.define(
            f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib, tags=tags
        )
        actual = self.ns().foo.default.tags
        self.assertTrue(isinstance(actual, list))
        self.assertEqual(actual, list(tags))

    def test_builtin_aten_ops_are_pt2_compliant(self):
        for op in [torch.ops.aten.sin.default, torch.ops.aten.sum.dim_IntList]:
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_builtin_torchscript_ops(self):
        for op in [torch.ops.aten.sub.complex, torch.ops.aten.mul.complex]:
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_autogen_aten_ops_are_pt2_compliant(self):
        for op in [
            torch.ops.aten.fill.Tensor_out,
        ]:
            self.assertIn(torch.Tag.generated, op.tags)
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_resolve_packet(self):
        x = torch.randn(3)
        result = torch._C._jit_resolve_packet("aten::sum", x)
        self.assertEqual(result, "default")

        result = torch._C._jit_resolve_packet("aten::sum", x, dim=1)
        self.assertEqual(result, "dim_IntList")

        with self.assertRaisesRegex(RuntimeError, "failed to match any schema"):
            result = torch._C._jit_resolve_packet("aten::sum", x, x, x)

    def test_define_bad_schema(self):
        lib = self.lib()
        with self.assertRaisesRegex(ValueError, "expected schema to look like"):
            torch.library.define(f"{self.test_ns}::foo", "foo(Tensor x) -> Tensor")

    def test_define_and_impl(self):
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        @torch.library.impl(f"{self.test_ns}::foo", "CPU", lib=lib)
        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        x = torch.randn(3)
        y = self.ns().foo(x)
        assert torch.allclose(y, x.sin())

    def test_define_validation(self):
        with self.assertRaisesRegex(ValueError, "namespace"):
            torch.library.define("foo", "(Tensor x) -> Tensor")

    def test_legacy_define(self):
        lib = self.lib()

        @torch.library.define(lib, "foo(Tensor x) -> Tensor")
        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        x = torch.randn(3)
        y = self.ns().foo(x)
        assert torch.allclose(y, x.sin())

    def test_impl_function(self):
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        torch.library.impl(f"{self.test_ns}::foo", "CPU", f, lib=lib)
        x = torch.randn(3)
        y = self.ns().foo(x)
        assert torch.allclose(y, x.sin())

    def test_legacy_impl(self):
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        @torch.library.impl(lib, "foo", "CPU")
        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        x = torch.randn(3)
        y = self.ns().foo(x)
        assert torch.allclose(y, x.sin())

    def test_defined_in_python(self):
        self.assertFalse(torch.ops.aten.sin.default._defined_in_python)
        self.assertFalse(torch.ops.aten.sum.dim_IntList._defined_in_python)

        lib = self.lib()
        torch.library.define("{self._test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)
        ns = self.ns()
        self.assertTrue(ns.foo.default._defined_in_python)

        torch.library.define(
            "{self._test_ns}::bar.overload", "(Tensor x) -> Tensor", lib=lib
        )
        self.assertTrue(ns.bar.overload._defined_in_python)

    def _test_impl_device(self, name, types, device):
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::{name}", "(Tensor x) -> Tensor", lib=lib)

        @torch.library.impl(f"{self.test_ns}::{name}", types)
        def f(x):
            x_np = x.cpu().numpy()
            y = torch.from_numpy(np.sin(x_np))
            return y.to(device=x.device)

        x = torch.randn(3, device=device)
        y = getattr(self.ns(), name)(x)
        assert torch.allclose(y, x.sin())

    def test_impl_device_cpu(self):
        self._test_impl_device("foo1", "default", "cpu")
        self._test_impl_device("foo2", ["cpu"], "cpu")
        self._test_impl_device("foo3", ["cpu", "cuda"], "cpu")

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_impl_device_cuda(self):
        self._test_impl_device("foo4", "default", "cuda")
        self._test_impl_device("foo5", ["cuda"], "cuda")
        self._test_impl_device("foo6", ["cpu", "cuda"], "cuda")

    def test_impl_device_function(self):
        lib = self.lib()
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        def f(x):
            x_np = x.cpu().numpy()
            y = torch.from_numpy(np.sin(x_np))
            return y.to(device=x.device)

        torch.library.impl(f"{self.test_ns}::foo", "default", f, lib=lib)
        x = torch.randn(3)
        y = self.ns().foo(x)
        assert torch.allclose(y, x.sin())

    def test_impl_device_invalid(self):
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu, cuda"):
            torch.library.impl("blah::blah", "somethingsomething")

    def test_autograd_function_backed_op(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

TORCH_LIBRARY(mylib, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        module = torch.utils.cpp_extension.load_inline(
            name="mylib",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        x = torch.ones(2, 2, requires_grad=True)
        temp = x.clone().detach()
        out = torch.ops.mylib.custom_op_backed_by_autograd_fn(x)
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad, temp)


def op_with_incorrect_schema(testcase, name):
    lib = testcase.lib()
    lib.define(f"{name}(Tensor x) -> Tensor")
    qualname = f"{testcase.test_ns}::{name}"
    lib.impl(name, lambda x: x[:], "CompositeExplicitAutograd")
    return testcase.get_op(qualname)


class MiniOpTest(CustomOpTestCaseBase):
    test_ns = "mini_op_test"

    def _init_op_delayed_backward_error(self):
        name = "delayed_error"
        qualname = f"{self.test_ns}::{name}"
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor")
        lib.impl(name, lambda x: x.clone(), "CompositeExplicitAutograd")
        op = self.get_op(qualname)

        class Op(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, grad):
                raise NotImplementedError

        def autograd_impl(x):
            return Op.apply(x)

        lib.impl(name, autograd_impl, "Autograd")
        return op

    def _init_op_with_no_abstract_impl(self):
        name = "no_abstract"
        qualname = f"{self.test_ns}::{name}"
        lib = self.lib()
        lib.define(f"{name}(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,))
        lib.impl(name, lambda x: x.clone(), "CPU")
        return torch._library.utils.lookup_op(qualname)

    def setUp(self):
        super().setUp()
        self._op_with_no_abstract_impl = self._init_op_with_no_abstract_impl()
        self._op_delayed_backward_error = self._init_op_delayed_backward_error()

    @optests.dontGenerateOpCheckTests("Testing this API")
    def test_dont_generate(self):
        op = op_with_incorrect_schema(self, "incorrect_schema")
        x = torch.randn(3)
        op(x)

    def test_mm(self):
        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(3, 5)
        result = torch.ops.aten.mm.default(x, y)
        self.assertEqual(result, x @ y)

    def test_mm_meta(self):
        x = torch.randn(2, 3, requires_grad=True, device="meta")
        y = torch.randn(3, 5, device="meta")
        result = torch.ops.aten.mm.default(x, y)
        self.assertEqual(result.shape, (x @ y).shape)

    def test_mm_fake(self):
        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn(2, 3, requires_grad=True, device="cpu")
            y = torch.randn(3, 5, device="cpu")
            result = torch.ops.aten.mm.default(x, y)
            self.assertEqual(result.shape, (x @ y).shape)

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
        op = self._op_with_no_abstract_impl
        x = torch.randn(3)
        op(x)

    def test_delayed_error(self):
        op = self._op_delayed_backward_error
        x = torch.randn([], requires_grad=True)
        y = op(x)
        with self.assertRaises(NotImplementedError):
            y.sum().backward()

    def test_delayed_error_no_requires_grad(self):
        op = self._op_delayed_backward_error
        x = torch.randn([])
        y = op(x)


class TestCustomOpAPI(TestCase):
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_basic(self):
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            x_np = x.numpy(force=True)
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = 3.14
        z = add(x, y)
        self.assertEqual(z, x + y)

        cpu_called = False

        @add.register_kernel("cpu")
        def _(x, y):
            nonlocal cpu_called
            cpu_called = True
            x_np = x.numpy()
            out_np = x_np + y
            return torch.from_numpy(out_np)

        z = add(x, y)
        self.assertEqual(z, x + y)
        self.assertTrue(cpu_called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_no_grad_skips_autograd(self):
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            x_np = x.numpy(force=True)
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        called = 0

        def setup_context(ctx, inputs, output):
            nonlocal called
            called += 1

        def backward(ctx, grad):
            raise AssertionError("should not be reached")

        add.register_autograd(backward, setup_context=setup_context)

        x = torch.randn(3, requires_grad=True)
        with torch.no_grad():
            y = add(x, 2.0)
        self.assertEqual(called, 0)
        self.assertEqual(y, x + 2.0)

        x.requires_grad_(False)
        y = add(x, 2.0)
        self.assertEqual(called, 0)
        self.assertEqual(y, x + 2.0)

        x = torch.randn(3, requires_grad=True)
        y = add(x, 2.0)
        self.assertEqual(called, 1)
        self.assertEqual(y, x + 2.0)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_manual_schema(self):
        @torch.library.custom_op(
            "_torch_testing::add",
            mutates_args=(),
            schema="(Tensor x, float y) -> Tensor",
        )
        def add(x, y):
            x_np = x.numpy(force=True)
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = 3.14
        z = add(x, y)
        self.assertEqual(z, x + y)

        @torch.library.custom_op(
            "_torch_testing::sin_",
            mutates_args=["x"],
            schema="(Tensor(a!) x) -> ()",
        )
        def sin_(x):
            x_np = x.numpy()
            np.sin(x_np, out=x_np)

        x = torch.randn(3)
        expected = x.sin()
        sin_(x)
        self.assertEqual(x, expected)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_kwarg_only_tensors(self):
        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo(x: Tensor, *, y: int, z: Tensor) -> Tensor:
                pass

        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo2(x: Tensor, *, y: int, z: Optional[Tensor]) -> Tensor:
                pass

        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):

            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo3(x: Tensor, *, y: int, z: List[Tensor]) -> Tensor:
                pass

        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, *, Tensor y) -> Tensor")
            with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):
                torch.library.register_autograd(
                    "_torch_testing::foo",
                    lambda grad: grad,
                    setup_context=lambda ctx, inputs, keyword_only_inputs, output: None,
                )

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_kwargonly_low_level(self):
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor x, *, float y) -> Tensor")
            called = False

            def foo_impl(x, *, y):
                return x * y

            lib.impl("foo", foo_impl, "CPU")

            def backward(ctx, grad):
                nonlocal called
                called = True
                return grad * ctx.y

            def setup_context(ctx, inputs, keyword_only_inputs, output):
                assert tuple(keyword_only_inputs.keys()) == ("y",)
                ctx.y = keyword_only_inputs["y"]

            torch.library.register_autograd(
                "_torch_testing::foo", backward, setup_context=setup_context, lib=lib
            )

            x = torch.randn(3, requires_grad=True)
            torch.ops._torch_testing.foo(x, y=3.14).sum().backward()
            self.assertTrue(called)
            self.assertEqual(x.grad, torch.tensor([3.14, 3.14, 3.14]))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_defaults(self):
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            lib.define("foo(Tensor w, int x = 2, *, int y = 3, int z) -> Tensor")

            def foo_impl(w, x=2, *, y=3, z):
                return w * x * y * z

            lib.impl("foo", foo_impl, "CPU")

            called = False

            def backward(ctx, grad):
                nonlocal called
                called = True
                return grad * ctx.c

            def setup_context(ctx, inputs, keyword_only_inputs, output):
                assert len(inputs) == 2
                assert inputs[1] == 2
                assert keyword_only_inputs == {"y": 3, "z": 42}
                ctx.c = keyword_only_inputs["y"] * keyword_only_inputs["z"] * inputs[1]

            torch.library.register_autograd(
                "_torch_testing::foo", backward, setup_context=setup_context, lib=lib
            )

            w = torch.randn(3, requires_grad=True)
            torch.ops._torch_testing.foo(w, z=42).sum().backward()
            self.assertTrue(called)
            self.assertEqual(w.grad, torch.full_like(w, 2 * 3 * 42))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_manual_schema_error(self):
        with self.assertRaisesRegex(ValueError, "the op mutates {'x'}"):

            @torch.library.custom_op(
                "_torch_testing::sin_",
                mutates_args=(),
                schema="(Tensor(a!) x) -> ()",
            )
            def sin_(x):
                x_np = x.numpy()
                np.sin(x_np, out=x_np)

    def test_supports_tensorlist(self):
        @torch._library.autograd.supports_tensorlist
        class Stack(torch.autograd.Function):
            @staticmethod
            def forward(ctx, xs):
                ctx.num_xs = len(xs)
                return torch.stack(xs)

            @staticmethod
            def backward(ctx, grad):
                expected = ([True] * ctx.num_xs,)
                self.assertEqual(ctx.needs_input_grad, expected)
                return list(grad.unbind(0))

        # call two applys, do a backward on the first
        def t():
            return torch.randn([], requires_grad=True)

        xs0 = [t(), t(), t()]
        xs1 = [t(), t(), t(), t()]
        y0 = Stack.apply(xs0)
        y1 = Stack.apply(xs1)
        grads = torch.autograd.grad(y0.sum(), xs0)
        self.assertEqual(grads, [torch.tensor(1.0) for _ in range(3)])

        # call one apply, do multiple backwards
        xs = [t(), t(), t()]
        y = Stack.apply(xs)
        _ = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        _ = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        grads = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        self.assertEqual(grads, [torch.tensor(1.0) for _ in range(3)])

        # error: on access forward, backward directly
        with self.assertRaisesRegex(NotImplementedError, "Function.forward directly"):
            Stack.forward(None, xs)
        with self.assertRaisesRegex(NotImplementedError, "Function.backward directly"):
            Stack.backward(None, xs)

        # the recursive case
        @torch._library.autograd.supports_tensorlist
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, xs):
                if len(xs) > 1:
                    return Foo.apply(xs[1:])
                ctx.len_xs = len(xs)
                return xs[0].sin()

            @staticmethod
            def backward(ctx, grad):
                result = [None] * ctx.len_xs
                result[-1] = grad.cos()
                return result

        # should work
        result = Foo.apply(xs)
        expected = xs[-1].sin()
        self.assertEqual(result, expected)

        # recursive on backward
        @torch._library.autograd.supports_tensorlist
        class Bar(torch.autograd.Function):
            @staticmethod
            def forward(ctx, xs):
                return [xs[i] + i for i in range(len(xs))]

            @staticmethod
            def backward(ctx, grads):
                f1 = Bar.apply(grads[:2])
                f2 = Bar.apply(grads[2:])
                return f1 + f2

        xs = [torch.tensor(0.0, requires_grad=True) for _ in range(5)]
        ys = Bar.apply(xs)
        sum(ys).backward()
        result = [xi.grad for xi in xs]
        self.assertEqual(result, torch.tensor([1.0, 2, 1, 2, 3]).unbind(0))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_default_values(self):
        defaults = []

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(
            x: Tensor,
            a: Optional[int] = None,
            b: float = 3.14,
            c: bool = True,
            d: int = 3,
            e: str = "foo",
            f: torch.dtype = torch.float,
            g: torch.dtype = torch.float32,
            h: torch.dtype = torch.int,
        ) -> Tensor:
            defaults.extend([a, b, c, d, e, f, g, h])
            return x.clone()

        x = torch.randn(3)
        f(x)
        self.assertEqual(
            defaults,
            [None, 3.14, True, 3, "foo", torch.float, torch.float32, torch.int],
        )

    def test_mutated_error(self):
        with self.assertRaisesRegex(
            ValueError, r".*{'y'} in mutates_args were not found"
        ):

            @torch.library.custom_op(
                "_torch_testing::numpy_sin_inplace",
                mutates_args={"y"},
                device_types="cpu",
            )
            def numpy_sin_inplace(x: Tensor) -> None:
                x_np = x.numpy()
                np.sin(x_np, out=x_np)

    def test_mutated(self):
        @torch.library.custom_op(
            "_torch_testing::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu"
        )
        def numpy_sin_inplace(x: Tensor) -> None:
            x_np = x.numpy()
            np.sin(x_np, out=x_np)

        x = torch.randn(3)
        version = x._version
        expected = x.sin()
        numpy_sin_inplace(x)
        self.assertEqual(x, expected)
        self.assertGreater(x._version, version)

        @torch.library.custom_op("_torch_testing::f", mutates_args={"y", "z", "w"})
        def f(
            x: Tensor, y: Optional[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            return

        x = torch.randn(3)
        y = torch.randn(3)
        z = [torch.randn(3), torch.randn(3)]
        w = [torch.randn(3), None, torch.randn(3)]
        initial_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )
        f(x, y, z, w)
        new_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )

        self.assertEqual(initial_versions[0], new_versions[0])
        initial_versions, _ = pytree.tree_flatten(initial_versions[1:])
        new_versions, _ = pytree.tree_flatten(new_versions[1:])
        for prev, after in zip(initial_versions, new_versions):
            if prev is None and after is None:
                continue
            self.assertGreater(after, prev)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @parametrize("idx", [0, 1, 2, 3, 4, 5])
    def test_library_register_fake_source(self, idx):
        opname = f"source{idx}"
        op = getattr(torch.ops._torch_testing, opname).default
        entry = torch._library.simple_registry.singleton.find(op._name)
        source = entry.fake_impl.kernel.source
        assert source is not None
        self.assertTrue("custom_op_db.py" in source)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_fake(self):
        for mode in ["function", "qualname", "opoverload"]:

            @torch.library.custom_op("_torch_testing::add", mutates_args=())
            def add(x: Tensor, y: float) -> Tensor:
                x_np = x.cpu().numpy()
                out_np = x_np + y
                return torch.from_numpy(out_np).to(x.device)

            called = False

            if mode == "function":
                dec = torch.library.register_fake(add)
                self.assertIsNotNone(dec)
            elif mode == "qualname":
                dec = torch.library.register_fake("_torch_testing::add")
                self.assertIsNotNone(dec)
            elif mode == "opoverload":
                dec = torch.library.register_fake(torch.ops._torch_testing.add.default)
                self.assertIsNotNone(dec)
            else:
                raise AssertionError("should not get here")

            @dec
            def _(x, y):
                nonlocal called
                called = True
                return torch.empty_like(x)

            with torch._subclasses.fake_tensor.FakeTensorMode():
                x = torch.randn(3)
                y = 3.14
                z = add(x, y)
                self.assertEqual(z.shape, x.shape)
                self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_kernel(self):
        modes = ["function", "qualname", "opoverload"]
        calls = ["decorator", "function"]
        device_types_options = ["cpu", None]

        for mode, call, device_types in itertools.product(
            modes, calls, device_types_options
        ):

            @torch.library.custom_op(
                "_torch_testing::add", mutates_args=(), device_types="cuda"
            )
            def add(x: Tensor, y: float) -> Tensor:
                x_np = x.cpu().numpy()
                out_np = x_np + y
                return torch.from_numpy(out_np).to(x.device)

            if mode == "function":
                op = add
            elif mode == "qualname":
                op = "_torch_testing::add"
            else:
                assert mode == "opoverload"
                op = torch.ops._torch_testing.add.default

            called = False

            if call == "decorator":

                @torch.library.register_kernel(op, device_types)
                def _(x, y):
                    nonlocal called
                    called = True
                    x_np = x.numpy()
                    out_np = x_np + y
                    return torch.from_numpy(out_np)

            else:
                assert call == "function"

                def add_cpu(x, y):
                    nonlocal called
                    called = True
                    x_np = x.numpy()
                    out_np = x_np + y
                    return torch.from_numpy(out_np)

                torch.library.register_kernel(op, device_types, add_cpu)

            x = torch.randn(3)
            y = 3.14
            z = add(x, y)
            self.assertEqual(z, x + y)
            self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_kernel_low_level(self):
        modes = ["qualname", "opoverload"]
        calls = ["decorator", "function"]
        device_types_options = [("cpu", "cuda"), "cpu", None]

        for mode, call, device_types in itertools.product(
            modes, calls, device_types_options
        ):
            with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                lib.define("add9(Tensor x, float y) -> Tensor")

                if mode == "qualname":
                    op = "_torch_testing::add9"
                else:
                    assert mode == "opoverload"
                    op = torch.ops._torch_testing.add9.default

                called = False

                if call == "decorator":

                    @torch.library.register_kernel(op, device_types, lib=lib)
                    def _(x, y):
                        nonlocal called
                        called = True
                        x_np = x.numpy()
                        out_np = x_np + y
                        return torch.from_numpy(out_np)

                else:
                    assert call == "function"

                    def add_cpu(x, y):
                        nonlocal called
                        called = True
                        x_np = x.numpy()
                        out_np = x_np + y
                        return torch.from_numpy(out_np)

                    torch.library.register_kernel(op, device_types, add_cpu, lib=lib)

                x = torch.randn(3)
                y = 3.14
                z = torch.ops._torch_testing.add9.default(x, y)
                self.assertEqual(z, x + y)
                self.assertTrue(called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_autograd(self):
        for mode in ["function", "qualname", "opoverload"]:

            @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
            def numpy_sin(x: Tensor) -> Tensor:
                x_np = x.cpu().numpy()
                y_np = np.sin(x_np)
                return torch.from_numpy(y_np).to(device=x.device)

            def setup_context(ctx, inputs, output) -> Tensor:
                (x,) = inputs
                ctx.save_for_backward(x)

            called = False

            def backward(ctx, grad):
                nonlocal called
                called = True
                (x,) = ctx.saved_tensors
                return grad * x.cos()

            if mode == "function":
                torch.library.register_autograd(
                    numpy_sin, backward, setup_context=setup_context
                )
            elif mode == "qualname":
                torch.library.register_autograd(
                    "mylib::numpy_sin", backward, setup_context=setup_context
                )
            elif mode == "opoverload":
                torch.library.register_autograd(
                    torch.ops.mylib.numpy_sin.default,
                    backward,
                    setup_context=setup_context,
                )

            x = torch.randn(3, requires_grad=True)
            y = numpy_sin(x)
            (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
            self.assertTrue(called)
            self.assertEqual(grad_x, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_autograd_low_level(self):
        for mode in ["qualname", "opoverload"]:
            with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                lib.define("sin5(Tensor x) -> Tensor")

                def numpy_sin(x: Tensor) -> Tensor:
                    x_np = x.cpu().detach().numpy()
                    y_np = np.sin(x_np)
                    return torch.from_numpy(y_np).to(device=x.device)

                def setup_context(ctx, inputs, output) -> Tensor:
                    (x,) = inputs
                    ctx.save_for_backward(x)

                called = False

                def backward(ctx, grad):
                    nonlocal called
                    called = True
                    (x,) = ctx.saved_tensors
                    return grad * x.cos()

                lib.impl("sin5", numpy_sin, "CPU")

                called = False

                if mode == "qualname":
                    torch.library.register_autograd(
                        "_torch_testing::sin5",
                        backward,
                        setup_context=setup_context,
                        lib=lib,
                    )
                elif mode == "opoverload":
                    torch.library.register_autograd(
                        torch.ops._torch_testing.sin5.default,
                        backward,
                        setup_context=setup_context,
                        lib=lib,
                    )
                x = torch.randn(3, requires_grad=True)
                y = torch.ops._torch_testing.sin5(x)
                (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
                self.assertTrue(called)
                self.assertEqual(grad_x, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_fake(self):
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            x_np = x.cpu().numpy()
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = 3.14
        z = add(x, y)
        self.assertEqual(z, x + y)

        try:
            with torch._subclasses.fake_tensor.FakeTensorMode():
                x = torch.randn(3)
                add(x, y)
            raise AssertionError("should not be hit")
        except RuntimeError as e:
            abstract_impl_error_msg = str(e)
        abstract_impl_error_msg = re.sub(
            r"0x.*>\)>", "0xDEADBEEF>)>", abstract_impl_error_msg
        ).replace(". ", ".\n")
        self.assertExpectedInline(
            abstract_impl_error_msg,
            """\
There was no fake impl registered for <CustomOpDef(_torch_testing::add)>.
This is necessary for torch.compile/export/fx tracing to work.
Please use `add.register_fake` to add an fake impl.""",
        )

        if not IS_WINDOWS:

            @torch.compile(backend="eager")
            def f(x, y):
                return add(x, y)

            x = torch.randn(3)
            with self.assertRaisesRegex(RuntimeError, "no fake impl"):
                f(x, y)

        abstract_called = False

        @add.register_fake
        def _(x, y):
            nonlocal abstract_called
            abstract_called = True
            return torch.empty_like(x)

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn(3)
            z = add(x, y)
            self.assertEqual(z.shape, x.shape)
            self.assertTrue(abstract_called)

    @skipIfTorchDynamo("recursive dynamo")
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work on windows")
    def test_compile(self):
        called_impl = False
        called_abstract = False

        @torch.library.custom_op("_torch_testing::linear", mutates_args=())
        def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            nonlocal called_impl
            called_impl = True
            x_np = x.numpy()
            w_np = weight.numpy()
            b_np = bias.numpy()
            out_np = np.add(x_np @ w_np.T, bias)
            return out_np

        @custom_linear.register_fake
        def _(x, weight, bias):
            nonlocal called_abstract
            called_abstract = True
            assert x.dim() == 2
            assert weight.dim() == 2
            assert bias.dim() == 1
            assert x.shape[1] == weight.shape[1]
            assert weight.shape[0] == bias.shape[0]
            assert x.device == weight.device
            return x.new_empty(x.size(0), weight.size(0))

        x = torch.randn(2, 2)
        weight = torch.randn(2, 2)
        bias = torch.randn(2)
        out = torch.compile(custom_linear, backend="eager", fullgraph=True)(
            x, weight, bias
        )
        self.assertEqual(out, torch.nn.functional.linear(x, weight, bias))
        self.assertTrue(called_impl)
        self.assertTrue(called_abstract)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_error_cases(self):
        @torch.library.custom_op("_torch_testing::g", mutates_args=())
        def g(x: Tensor) -> Tensor:
            return x.sin()

        x = torch.randn(3, requires_grad=True)
        y = g(x)
        with self.assertRaisesRegex(RuntimeError, "no autograd formula"):
            y.sum().backward()

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_replacement(self):
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            return x.sin()

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x.sin())

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            return x.cos()

        y = f(x)
        self.assertEqual(y, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_split_device(self):
        cpu_call_count = 0
        cuda_call_count = 0

        @torch.library.custom_op(
            "_torch_testing::f", mutates_args=(), device_types="cpu"
        )
        def f(x: Tensor) -> Tensor:
            nonlocal cpu_call_count
            cpu_call_count += 1
            x_np = x.numpy()
            out_np = np.sin(x_np)
            return torch.from_numpy(out_np)

        @f.register_kernel("cuda")
        def _(x: Tensor) -> Tensor:
            nonlocal cuda_call_count
            cuda_call_count += 1
            x_np = x.cpu().numpy()
            out_np = np.sin(x_np)
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x.sin())
        self.assertEqual(cpu_call_count, 1)
        self.assertEqual(cuda_call_count, 0)

        x = x.cuda()
        y = f(x)
        self.assertEqual(y, x.sin())
        self.assertEqual(cpu_call_count, 1)
        self.assertEqual(cuda_call_count, 1)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_multi_types(self):
        @torch.library.custom_op(
            "_torch_testing::f", mutates_args=(), device_types=("cpu", "cuda")
        )
        def f(x: Tensor) -> Tensor:
            x_np = x.cpu().numpy()
            out_np = np.sin(x_np)
            return torch.from_numpy(out_np).to(x.device)

        x = torch.randn(3)
        y = f(x)
        self.assertEqual(y, x.sin())
        x = x.cuda()
        y = f(x)
        self.assertEqual(y, x.sin())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_overloading(self):
        called_f = 0
        called_f1 = 0

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            nonlocal called_f
            called_f += 1
            return x.clone()

        x = torch.randn(2, 3)
        torch.ops._torch_testing.f(x)
        self.assertEqual(called_f, 1)

        @torch.library.custom_op("_torch_testing::f.overload", mutates_args=())
        def f1(x: Tensor, y: Tensor) -> Tensor:
            nonlocal called_f1
            called_f1 += 1
            return x.clone()

        torch.ops._torch_testing.f(x, x)
        self.assertEqual(called_f1, 1)

    def test_disallows_output_aliasing(self):
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            return x.view(-1)

        x = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            f(x)

        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            return x

        x = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            f(x)

        @torch.library.custom_op(
            "_torch_testing::f", mutates_args={"x"}, device_types="cpu"
        )
        def numpy_sin_inplace(x: Tensor) -> Tensor:
            x_np = x.numpy()
            np.sin(x_np, out=x_np)
            return x

        x = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            numpy_sin_inplace(x)


class MiniOpTestOther(CustomOpTestCaseBase):
    test_ns = "mini_op_test"

    def test_nonzero_again(self):
        x = torch.tensor([0, 1, 2, 0, 0])
        y = torch.ops.aten.nonzero.default(x)
        self.assertEqual(y, torch.tensor([[1], [2]]))


optests.generate_opcheck_tests(
    MiniOpTest,
    ["aten", "mini_op_test"],
    get_file_path_2(
        os.path.dirname(__file__),
        "minioptest_failures_dict.json",
    ),
    additional_decorators={
        "test_pt2_compliant_tag_mini_op_test_no_abstract": [unittest.expectedFailure]
    },
    test_utils=optests.generate_tests.DEPRECATED_DEFAULT_TEST_UTILS,
)

optests.generate_opcheck_tests(
    MiniOpTestOther,
    ["aten", "mini_op_test"],
    get_file_path_2(
        os.path.dirname(__file__),
        "minioptest_failures_dict.json",
    ),
    test_utils=optests.generate_tests.DEPRECATED_DEFAULT_TEST_UTILS,
)


class TestGenerateOpcheckTests(CustomOpTestCaseBase):
    def test_MiniOpTest(self):
        for orig_test in ["test_mm", "test_nonzero"]:
            for (
                test
            ) in torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS:
                expected_test = f"{test}__{orig_test}"
                self.assertTrue(hasattr(MiniOpTest, expected_test), msg=expected_test)

    def test_generate_repro_save_data(self):
        from torch.testing._internal.optests.generate_tests import generate_repro

        args = (torch.ones(2, 2),)
        kwargs = {"mat2": torch.zeros(2, 2)}
        actual = generate_repro(
            "test_schema",
            torch.ops.aten.sin.default,
            args,
            kwargs,
            save_data=True,
            dry_run=True,
        )
        actual = re.sub(r"torch.load\(\".*\.pt\"\)", 'torch.load("repro.pt")', actual)
        self.assertExpectedInline(
            actual,
            """\
# =========================================================
# BEGIN REPRO SCRIPT
# =========================================================
import torch
from torch.testing._internal.optests import opcheck

# Make sure you have loaded the library that contains the op
# via an import or torch.ops.load_library(...)
op = torch.ops.aten.sin.default

args, kwargs = torch.load("repro.pt")
opcheck(op, args, kwargs, test_utils="test_schema")
# =========================================================
# END REPRO SCRIPT
# =========================================================
""",
        )

    def test_generate_repro_no_save_data(self):
        from torch.testing._internal.optests.generate_tests import generate_repro

        args = (torch.ones(2, 2),)
        kwargs = {"mat2": torch.zeros(2, 2)}
        actual = generate_repro(
            "test_schema",
            torch.ops.aten.sin.default,
            args,
            kwargs,
            save_data=False,
            dry_run=True,
        )
        self.assertExpectedInline(
            actual,
            """\
# =========================================================
# BEGIN REPRO SCRIPT
# =========================================================
import torch
from torch.testing._internal.optests import opcheck

# Make sure you have loaded the library that contains the op
# via an import or torch.ops.load_library(...)
op = torch.ops.aten.sin.default

# If you rerun your test with PYTORCH_OPCHECK_PRINT_BETTER_REPRO=1
# we will fill them in same (args, kwargs) as in your test
args = ()  # args to the operator
kwargs = {}  # kwargs to the operator
opcheck(op, args, kwargs, test_utils="test_schema")
# =========================================================
# END REPRO SCRIPT
# =========================================================
""",
        )

    def test_failures_dict_validation(self):
        from torch.testing._internal.optests.generate_tests import (
            FailuresDict,
            validate_failures_dict_structure,
        )

        failures = {
            "mini_op_test::incorrect_schema": {
                "MiniOpTest.test_aot_dispatch_dynamic__test_delayed_error": {
                    "comment": "",
                    "status": "success",
                }
            }
        }
        with self.assertRaisesRegex(RuntimeError, "got status=success"):
            validate_failures_dict_structure(
                FailuresDict("", failures),
                torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
                MiniOpTest,
            )

        failures = {
            "mini_op_test::incorrect_schema": {
                "MiniOpTest.test_aot_dispatch__test_delayed_error": {
                    "comment": "",
                    "status": "xfail",
                },
            }
        }
        with self.assertRaisesRegex(RuntimeError, "should begin with one of"):
            validate_failures_dict_structure(
                FailuresDict("", failures),
                torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
                MiniOpTest,
            )

        failures = {
            "mini_op_test::incorrect_schema": {
                "MiniOpTest.test_aot_dispatch_dynamic__test_delayed_error_nopenopenope": {
                    "comment": "",
                    "status": "xfail",
                },
            }
        }
        with self.assertRaisesRegex(RuntimeError, "does not exist on the TestCase"):
            validate_failures_dict_structure(
                FailuresDict("", failures),
                torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
                MiniOpTest,
            )

    def test_dont_generate_decorator(self):
        self.assertTrue(hasattr(MiniOpTest, "test_dont_generate"))
        self.assertFalse(hasattr(MiniOpTest, "test_schema__test_dont_generate"))

    def test_opcheck(self):
        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "OpOverload"):
            torch.library.opcheck(torch.sin, (x,))
        with self.assertRaisesRegex(ValueError, "test_utils to be subset of"):
            torch.library.opcheck(torch.ops.aten.sin.default, (x,), test_utils="blah")
        result = torch.library.opcheck(torch.ops.aten.sin.default, (x,))

        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )

        result = torch.library.opcheck(
            torch.ops.aten.sin.default, (x,), test_utils="test_schema"
        )
        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
            },
        )

        result = torch.library.opcheck(
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

    def test_opcheck_customopdef(self):
        sample_inputs = [
            (torch.randn(3),),
            (torch.randn(3, requires_grad=True),),
        ]
        if torch.cuda.is_available():
            sample_inputs.extend(
                [
                    (torch.randn(3, device="cuda"),),
                    (torch.randn(3, device="cuda", requires_grad=True),),
                ]
            )
        for args in sample_inputs:
            torch.library.opcheck(custom_op_db.numpy_cube, args)

    def test_is_inside_opcheck_mode(self):
        self.assertFalse(optests.is_inside_opcheck_mode())
        with optests.generate_tests.OpCheckMode(
            ["foo"], "bar", lambda x: x, None, "baz", "brr"
        ):
            self.assertTrue(optests.is_inside_opcheck_mode())

    def test_opcheck_bad_op(self):
        op = op_with_incorrect_schema(self, "foo")
        x = torch.randn(3)
        with self.assertRaisesRegex(Exception, "is not defined to alias output"):
            torch.library.opcheck(op, (x,))

        result = torch.library.opcheck(op, (x,), raise_exception=False)
        self.assertTrue(isinstance(result["test_schema"], RuntimeError))
        del result["test_schema"]
        self.assertEqual(
            result,
            {
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )

    def test_opcheck_does_not_require_extra_deps(self):
        # torch.testing._internal.common_utils comes with a lot of additional
        # test-time dependencies. Since opcheck is public API, it should be
        # usable only with pytorch install-time dependencies.
        cmd = [
            sys.executable,
            "-c",
            "import torch; import sys; \
               x = torch.randn(3, requires_grad=True); \
               torch.library.opcheck(torch.ops.aten.sin.default, (x,)); \
               assert 'expecttest' not in sys.modules; \
               assert 'torch.testing._internal.common_utils' not in sys.modules",
        ]
        subprocess.check_output(cmd, shell=False)


class TestTypeConversion(TestCase):
    """In infer_schema(), we try to suggest a correct type when the type annotation is wrong."""

    def setUp(self):
        self.supported_base_types = [
            int,
            float,
            bool,
            str,
            torch.device,
            torch.Tensor,
            torch.dtype,
            torch.types.Number,
        ]

    def test_simple_tuple(self):
        self.assertEqual(List, tuple_to_list(Tuple))

    def test_supported_types(self):
        for t in self.supported_base_types:
            result_type = tuple_to_list(Tuple[t, t, t])
            self.assertEqual(result_type, List[t])

            result_type = tuple_to_list(Tuple[t])
            self.assertEqual(result_type, List[t])

    def test_optional(self):
        for t in self.supported_base_types:
            result_type = tuple_to_list(Tuple[t, Optional[t]])
            self.assertEqual(result_type, List[Optional[t]])

            result_type = tuple_to_list(Tuple[t, t, Optional[t]])
            self.assertEqual(result_type, List[Optional[t]])

            result_type = tuple_to_list(Tuple[t, ...])
            self.assertEqual(result_type, List[t])

    def test_mixed_types(self):
        result_type = tuple_to_list(Tuple[int, float])
        self.assertEqual(result_type, List[typing.Union[int, float]])

        result_type = tuple_to_list(Tuple[int, float, str])
        self.assertEqual(result_type, List[typing.Union[int, float, str]])


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestCustomOpTesting, globals(), only_for=only_for)
instantiate_parametrized_tests(TestCustomOp)
instantiate_parametrized_tests(TestCustomOpAPI)

if __name__ == "__main__":
    run_tests()
