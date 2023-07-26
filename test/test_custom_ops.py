# Owner(s): ["module: custom-operators"]

from torch.testing._internal.common_utils import *  # noqa: F403
from torch.testing._internal.common_device_type import *  # noqa: F403
from torch._custom_op.impl import custom_op
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.optests.compile_check import operator_compile_check


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


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestCustomOpTesting, globals(), only_for=only_for)

if __name__ == "__main__":
    run_tests()
