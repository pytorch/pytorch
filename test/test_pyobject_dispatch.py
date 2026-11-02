# Owner(s): ["module: custom-operators"]

import unittest
from typing import cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class PyObjectDispatchReturnsTest(unittest.TestCase):
    def test_custom_op_normalizes_tensor_list_and_mixed_returns(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_returns::tensor_list", mutates_args=()
        )
        def tensor_list(x: torch.Tensor) -> list[torch.Tensor]:
            return cast(list[torch.Tensor], (x.clone(), x + 1))

        @torch.library.custom_op(
            "_pyobject_dispatch_returns::mixed", mutates_args=()
        )
        def mixed(x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
            return (tensor_list(x), x.clone())

        def backward(ctx, grad_outputs):
            self.assertIsInstance(grad_outputs, list)
            return (grad_outputs[0] + grad_outputs[1],)

        tensor_list.register_autograd(
            backward, setup_context=lambda ctx, inputs, output: None
        )

        self.assertTrue(
            torch.ops._pyobject_dispatch_returns.tensor_list.default._is_pyobj_dispatcher_enabled()
        )
        self.assertTrue(
            torch.ops._pyobject_dispatch_returns.mixed.default._is_pyobj_dispatcher_enabled()
        )

        x = torch.ones(2)
        result = tensor_list(x)
        self.assertIsInstance(result, list)
        torch.testing.assert_close(result, [x, x + 1])

        values, other = mixed(x)
        self.assertIsInstance(values, list)
        torch.testing.assert_close(values, [x, x + 1])
        torch.testing.assert_close(other, x)

        x = torch.ones(2, requires_grad=True)
        sum(t.sum() for t in tensor_list(x)).backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2))

    def test_custom_op_normalizes_scalar_and_void_returns(self) -> None:
        calls = []

        @torch.library.custom_op(
            "_pyobject_dispatch_returns::scalar", mutates_args=()
        )
        def scalar(x: torch.Tensor) -> float:
            return float(x.numel())

        @torch.library.custom_op(
            "_pyobject_dispatch_returns::void", mutates_args=()
        )
        def void(x: torch.Tensor) -> None:
            calls.append(x.numel())

        x = torch.ones(2)
        self.assertEqual(scalar(x), 2.0)
        self.assertIsNone(void(x))
        self.assertEqual(calls, [2])

    def test_library_op_normalizes_optional_and_multiple_returns(self) -> None:
        lib = torch.library.Library("_pyobject_dispatch_returns_library", "DEF")
        self.addCleanup(lib._destroy)
        lib.define("optional_some(Tensor x) -> Tensor?")
        lib.define("optional_none(Tensor x) -> Tensor?")
        lib.define("multiple(Tensor x) -> (Tensor, Tensor)")
        lib.impl("optional_some", lambda x: x.clone(), "CPU")
        lib.impl("optional_none", lambda x: None, "CPU")
        lib.impl("multiple", lambda x: [x.clone(), x + 1], "CPU")
        lib.impl("optional_some", torch.library.fallthrough_kernel, "Autograd")
        lib.impl("optional_none", torch.library.fallthrough_kernel, "Autograd")
        lib.impl("multiple", torch.library.fallthrough_kernel, "Autograd")

        optional_some = (
            torch.ops._pyobject_dispatch_returns_library.optional_some.default
        )
        optional_none = (
            torch.ops._pyobject_dispatch_returns_library.optional_none.default
        )
        multiple = torch.ops._pyobject_dispatch_returns_library.multiple.default
        optional_some._enable_pyobj_dispatch(True)
        optional_none._enable_pyobj_dispatch(True)
        multiple._enable_pyobj_dispatch(True)

        x = torch.ones(2)
        torch.testing.assert_close(optional_some(x), x)
        self.assertIsNone(optional_none(x))
        first, second = multiple(x)
        torch.testing.assert_close(first, x)
        torch.testing.assert_close(second, x + 1)

    def test_invalid_void_return_reports_schema_mismatch(self) -> None:
        lib = torch.library.Library("_pyobject_dispatch_invalid_returns", "DEF")
        self.addCleanup(lib._destroy)
        lib.define("void(Tensor x) -> ()")
        lib.impl("void", lambda x: x, "CPU")
        lib.impl("void", torch.library.fallthrough_kernel, "Autograd")
        op = torch.ops._pyobject_dispatch_invalid_returns.void.default
        op._enable_pyobj_dispatch(True)

        with self.assertRaisesRegex(RuntimeError, "return None"):
            op(torch.ones(2))


class PyObjectDispatchArgumentsTest(unittest.TestCase):
    def test_custom_op_accepts_keywords_without_keyword_only_args(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_no_kwonly::scale", mutates_args=()
        )
        def scale(x: torch.Tensor, multiplier: float = 2.0) -> torch.Tensor:
            return x * multiplier

        x = torch.ones(2)
        op = torch.ops._pyobject_dispatch_no_kwonly.scale.default
        self.assertTrue(op._is_pyobj_dispatcher_enabled())
        torch.testing.assert_close(scale(x=x, multiplier=3.0), x * 3)
        torch.testing.assert_close(scale(x, multiplier=4.0), x * 4)
        torch.testing.assert_close(scale(x=x), x * 2)

        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        torch.testing.assert_close(
            op.redispatch(keyset, x=x, multiplier=5.0), x * 5
        )

    def test_custom_op_accepts_keyword_only_defaults(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_args_custom::scale", mutates_args=()
        )
        def scale(
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
            *,
            multiplier: float = 2.0,
        ) -> torch.Tensor:
            return x * multiplier + (bias if bias is not None else 0)

        x = torch.ones(2)
        op = torch.ops._pyobject_dispatch_args_custom.scale.default
        self.assertTrue(op._is_pyobj_dispatcher_enabled())
        torch.testing.assert_close(scale(x), x * 2)
        torch.testing.assert_close(scale(x=x, bias=x), x * 3)

    def test_kwargs_defaults_tensor_lists_and_scalars(self) -> None:
        lib = torch.library.Library("_pyobject_dispatch_args", "DEF")
        self.addCleanup(lib._destroy)
        lib.define(
            "weighted(Tensor[] tensors, float[] weights, float scale=1.0, "
            "*, Tensor? bias=None) -> Tensor"
        )
        received = []

        def weighted(tensors, weights, scale=1.0, *, bias=None):
            received.append((type(tensors), type(weights), type(scale), bias))
            result = sum(t * w for t, w in zip(tensors, weights)) * scale
            return result + bias if bias is not None else result

        lib.impl("weighted", weighted, "CPU")
        lib.impl("weighted", torch.library.fallthrough_kernel, "Autograd")
        op = torch.ops._pyobject_dispatch_args.weighted.default
        op._enable_pyobj_dispatch(True)

        x = torch.ones(2)
        expected = x * 4
        result = op((x, x * 3), (0.5, 0.5), scale=2, bias=x)
        torch.testing.assert_close(result, expected + x)
        self.assertEqual(received[-1][:3], (list, list, float))
        self.assertIs(received[-1][3], x)

        result = op(weights=[0.5, 0.5], tensors=[x, x * 3])
        torch.testing.assert_close(result, x * 2)
        self.assertEqual(received[-1][:3], (list, list, float))
        self.assertIsNone(received[-1][3])

        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        result = op.redispatch(
            keyset, weights=(0.5, 0.5), tensors=(x, x * 3), bias=x
        )
        torch.testing.assert_close(result, x * 3)
        self.assertIs(received[-1][3], x)

        op._enable_pyobj_dispatch(False)
        torch.testing.assert_close(
            op(weights=(0.5, 0.5), tensors=(x, x * 3)), x * 2
        )
        op._enable_pyobj_dispatch(True)

        with self.assertRaises((RuntimeError, TypeError)):
            op((x,), (1.0,), unknown=x)

    def test_ordered_symints_and_out_of_order_scalar_kwargs(self) -> None:
        lib = torch.library.Library("_pyobject_dispatch_scalar_args", "DEF")
        self.addCleanup(lib._destroy)
        lib.define(
            "combine(Tensor x, SymInt n, float scale, bool negate, "
            "Tensor? bias=None) -> (Tensor, Tensor, Tensor)"
        )
        received = []

        def combine(x, n, scale, negate, bias=None):
            received.append((type(n), type(scale), type(negate), bias))
            result = x * n * scale
            if negate:
                result = -result
            if bias is not None:
                result = result + bias
            return result, result + 1, x.clone()

        lib.impl("combine", combine, "CPU")
        lib.impl("combine", torch.library.fallthrough_kernel, "Autograd")
        op = torch.ops._pyobject_dispatch_scalar_args.combine.default
        op._enable_pyobj_dispatch(True)
        x = torch.ones(2)

        result = op(x=x, n=2, scale=1.5, negate=False, bias=x)
        torch.testing.assert_close(result, (x * 4, x * 4 + 1, x))
        self.assertEqual(received[-1][:3], (int, float, bool))
        self.assertIs(received[-1][3], x)

        result = op(scale=2, negate=True, x=x, n=3)
        torch.testing.assert_close(result, (-x * 6, -x * 6 + 1, x))
        self.assertEqual(received[-1][:3], (int, float, bool))
        self.assertIsNone(received[-1][3])

        class IntSubclass(int):
            pass

        result = op(x=x, n=IntSubclass(2), scale=1.0, negate=False)
        torch.testing.assert_close(result, (x * 2, x * 2 + 1, x))
        self.assertEqual(received[-1][0], int)

        with self.assertRaises((RuntimeError, TypeError, OverflowError)):
            op(x=x, n=1 << 80, scale=1.0, negate=False)

    def test_string_device_argument_is_normalized(self) -> None:
        lib = torch.library.Library("_pyobject_dispatch_device_arg", "DEF")
        self.addCleanup(lib._destroy)
        lib.define("factory(Device device) -> Tensor")
        received = []

        def factory(device):
            received.append(device)
            return torch.ones(2, device=device)

        lib.impl("factory", factory, "CompositeExplicitAutograd")
        op = torch.ops._pyobject_dispatch_device_arg.factory.default
        op._enable_pyobj_dispatch(True)

        torch.testing.assert_close(op("cpu"), torch.ones(2))
        self.assertIsInstance(received[0], torch.device)

    def test_dtype_and_device_kwargs_are_normalized(self) -> None:
        lib = torch.library.Library("_pyobject_dispatch_type_args", "DEF")
        self.addCleanup(lib._destroy)
        lib.define("cast(Tensor x, *, ScalarType dtype, Device device) -> Tensor")
        received = []

        def cast(x, *, dtype, device):
            received.append((dtype, device))
            return x.to(dtype=dtype, device=device)

        lib.impl("cast", cast, "CPU")
        lib.impl("cast", torch.library.fallthrough_kernel, "Autograd")
        op = torch.ops._pyobject_dispatch_type_args.cast.default
        op._enable_pyobj_dispatch(True)

        result = op(torch.ones(2), device="cpu", dtype=torch.float64)
        torch.testing.assert_close(result, torch.ones(2, dtype=torch.float64))
        self.assertIs(received[0][0], torch.float64)
        self.assertIsInstance(received[0][1], torch.device)


class PyObjectDispatchPythonKeyTest(unittest.TestCase):
    def test_mode_normalizes_tensor_list_return(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_python_key::list_return", mutates_args=()
        )
        def custom(x: torch.Tensor) -> list[torch.Tensor]:
            return cast(list[torch.Tensor], (x.clone(), x + 1))

        op = torch.ops._pyobject_dispatch_python_key.list_return.default
        calls = []

        class TrackingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is op:
                    calls.append(func)
                return func(*args, **(kwargs or {}))

        x = torch.ones(2)
        with TrackingMode():
            result = custom(x)
        self.assertIsInstance(result, list)
        torch.testing.assert_close(result, [x, x + 1])
        self.assertEqual(calls, [op])

    def test_torch_dispatch_mode_intercepts_custom_op(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_python_key::mode", mutates_args=()
        )
        def custom(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        op = torch.ops._pyobject_dispatch_python_key.mode.default
        calls = []

        class TrackingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is op:
                    calls.append(func)
                return func(*args, **(kwargs or {}))

        x = torch.ones(2)
        with TrackingMode():
            result = custom(x)
        torch.testing.assert_close(result, x)
        self.assertEqual(calls, [op])

    def test_tensor_subclass_intercepts_custom_op(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_python_key::subclass", mutates_args=()
        )
        def custom(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        op = torch.ops._pyobject_dispatch_python_key.subclass.default
        calls = []

        class TrackingTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @staticmethod
            def __new__(cls, value):
                return torch.Tensor._make_subclass(cls, value)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func is op:
                    calls.append(func)
                with torch._C._DisableTorchDispatch():
                    return func(*args, **(kwargs or {}))

        x = TrackingTensor(torch.ones(2))
        result = custom(x)
        torch.testing.assert_close(result, torch.ones(2))
        self.assertEqual(calls, [op])

    def test_mode_precedes_subclass_with_boxed_parity(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_python_key::precedence", mutates_args=()
        )
        def custom(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        op = torch.ops._pyobject_dispatch_python_key.precedence.default
        calls = []

        class TrackingMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is op:
                    calls.append("mode")
                return func(*args, **(kwargs or {}))

        class TrackingTensor(torch.Tensor):
            __torch_function__ = torch._C._disabled_torch_function_impl

            @staticmethod
            def __new__(cls, value):
                return torch.Tensor._make_subclass(cls, value)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func is op:
                    calls.append("subclass")
                with torch._C._DisableTorchDispatch():
                    return func(*args, **(kwargs or {}))

        x = TrackingTensor(torch.ones(2))
        with TrackingMode():
            result = custom(x)
        torch.testing.assert_close(result, torch.ones(2))
        self.assertEqual(calls, ["mode", "subclass"])

        calls.clear()
        op._enable_pyobj_dispatch(False)
        with TrackingMode():
            boxed_result = custom(x)
        torch.testing.assert_close(boxed_result, result)
        self.assertEqual(calls, ["mode", "subclass"])


class PyObjectDispatchProfileTest(unittest.TestCase):
    def test_pyobj_dispatched_op_appears_in_profiler(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_profile::visible", mutates_args=()
        )
        def visible(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        op = torch.ops._pyobject_dispatch_profile.visible.default
        self.assertTrue(op._is_pyobj_dispatcher_enabled())
        x = torch.ones(2)
        with torch.profiler.profile() as prof:
            for _ in range(3):
                visible(x)
        names = {e.key for e in prof.key_averages()}
        self.assertIn("_pyobject_dispatch_profile::visible", names)


class PyObjectDispatchCachedDefaultsTest(unittest.TestCase):
    def test_omitted_scalar_defaults_use_cached_values(self) -> None:
        @torch.library.custom_op(
            "_pyobject_dispatch_defaults::scaled", mutates_args=()
        )
        def scaled(
            x: torch.Tensor,
            scale: float = 2.0,
            count: int = 3,
            flag: bool = True,
            other: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = x * scale * count
            if flag:
                out = out + 1
            if other is not None:
                out = out + other
            return out

        x = torch.ones(2)
        for _ in range(3):
            torch.testing.assert_close(scaled(x), torch.full((2,), 7.0))
        torch.testing.assert_close(
            scaled(x, count=1, other=x), torch.full((2,), 4.0)
        )
        torch.testing.assert_close(
            scaled(x=x, flag=False, scale=1.0), torch.full((2,), 3.0)
        )
