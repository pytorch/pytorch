# Owner(s): ["module: custom-operators"]

import unittest
from typing import cast

import torch


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
