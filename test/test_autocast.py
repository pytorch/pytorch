# Owner(s): ["module: unknown"]

import unittest

import torch
from torch.testing._internal.autocast_test_lists import (
    TestAutocast,
    get_autocast_test_lists,
)
from torch.testing._internal.common_device_type import (
    DeviceTypeTestBase,
    instantiate_device_type_tests,
    onlyAccelerator,
    skipMPS,
)
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils._python_dispatch import TorchDispatchMode


class TestAutocastOps(TestAutocast, DeviceTypeTestBase):
    """Device-agnostic autocast op-level tests.

    Runs cast-policy tests (low-precision, fp32, promote) for all devices
    that have a registered test list in autocast_test_lists.py.
    New backends register via register_autocast_test_lists() to get these
    tests automatically.
    """

    def setUp(self):
        super().setUp()
        dev = torch.device(self.device_type)
        self.autocast_lists = get_autocast_test_lists(self.device_type, dev)
        if self.autocast_lists is None:
            self.skipTest(
                f"No autocast test lists registered for {self.device_type}"
            )
        self.amp_dtype = torch.get_autocast_dtype(device_type=self.device_type)

    def tearDown(self):
        if hasattr(self, "autocast_lists") and self.autocast_lists is not None:
            del self.autocast_lists
        super().tearDown()

    @skipIfTorchDynamo()
    def test_autocast_torch_expect_builtin_promote(self, device):
        for item in self.autocast_lists.torch_expect_builtin_promote:
            if len(item) == 4:
                op, args1, args2, out_type = item
                self._run_autocast_outofplace(
                    op, args1, torch.float32, device=self.device_type,
                    out_type=out_type
                )
                self._run_autocast_outofplace(
                    op, args2, torch.float32, device=self.device_type,
                    out_type=out_type, amp_dtype=torch.float16,
                )
            else:
                op, args, out_type = item
                self._run_autocast_outofplace(
                    op, args, torch.float32, device=self.device_type,
                    out_type=out_type, amp_dtype=self.amp_dtype,
                )

    @skipIfTorchDynamo()
    def test_autocast_methods_expect_builtin_promote(self, device):
        if not hasattr(self.autocast_lists, "methods_expect_builtin_promote"):
            self.skipTest("No methods_expect_builtin_promote list")
        for item in self.autocast_lists.methods_expect_builtin_promote:
            if len(item) == 4:
                op, args1, args2, out_type = item
                self._run_autocast_outofplace(
                    op, args1, torch.float32, device=self.device_type,
                    module=None, out_type=out_type,
                )
                self._run_autocast_outofplace(
                    op, args2, torch.float32, device=self.device_type,
                    module=None, out_type=out_type, amp_dtype=torch.float16,
                )
            else:
                op, args, out_type = item
                self._run_autocast_outofplace(
                    op, args, torch.float32, device=self.device_type,
                    module=None, out_type=out_type, amp_dtype=self.amp_dtype,
                )

    @skipIfTorchDynamo()
    def test_autocast_torch_low_precision(self, device):
        if not hasattr(self.autocast_lists, "torch_16"):
            self.skipTest("No torch_16 list")
        for op_with_args in self.autocast_lists.torch_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, self.amp_dtype, device=self.device_type,
                add_kwargs=maybe_kwargs, amp_dtype=self.amp_dtype,
            )

    @skipIfTorchDynamo()
    def test_autocast_nn_low_precision(self, device):
        if not hasattr(self.autocast_lists, "nn_16"):
            self.skipTest("No nn_16 list")
        for op_with_args in self.autocast_lists.nn_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, self.amp_dtype, device=self.device_type,
                module=torch._C._nn, add_kwargs=maybe_kwargs,
                amp_dtype=self.amp_dtype,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_fp32(self, device):
        if not hasattr(self.autocast_lists, "torch_fp32"):
            self.skipTest("No torch_fp32 list")
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, torch.float32, device=self.device_type,
                add_kwargs=maybe_kwargs, amp_dtype=self.amp_dtype,
            )

    @skipIfTorchDynamo()
    def test_autocast_nn_fp32(self, device):
        if not hasattr(self.autocast_lists, "nn_fp32"):
            self.skipTest("No nn_fp32 list")
        for op_with_args in self.autocast_lists.nn_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, torch.float32, device=self.device_type,
                module=torch._C._nn, add_kwargs=maybe_kwargs,
                amp_dtype=self.amp_dtype,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_need_autocast_promote(self, device):
        if not hasattr(self.autocast_lists, "torch_need_autocast_promote"):
            self.skipTest("No torch_need_autocast_promote list")
        for item in self.autocast_lists.torch_need_autocast_promote:
            if len(item) == 3:
                op, args1, args2 = item
                self._run_autocast_outofplace(
                    op, args1, torch.float32, device=self.device_type
                )
                self._run_autocast_outofplace(
                    op, args2, torch.float32, device=self.device_type,
                    amp_dtype=torch.float16,
                )
            else:
                op, args = item
                self._run_autocast_outofplace(
                    op, args, torch.float32, device=self.device_type,
                    amp_dtype=self.amp_dtype,
                )

    def test_autocast_rnn(self, device):
        if self.device_type != "cpu":
            self.skipTest("RNN autocast test is CPU-specific (MKLDNN)")
        if not (
            torch.backends.mkldnn.is_available()
            and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        ):
            self.skipTest("MKLDNN bf16 not supported")

        x = torch.randn(1, 2, 1)
        hx = torch.randn(2, 2, 1)
        cx = torch.randn(2, 2, 1)

        m = torch.nn.LSTM(1, 1, 2).to(torch.bfloat16)

        with self.assertRaisesRegex(
            ValueError, r"RNN input dtype .* does not match weight dtype"
        ):
            m(x, (hx, cx))

        with torch.amp.autocast(device_type="cpu"):
            m(x, (hx, cx))

    def test_autocast_disabled_with_fp32_dtype(self, device):
        with torch.autocast(
            device_type=self.device_type, dtype=torch.float32, enabled=False
        ):
            _ = torch.ones(10)

    def test_generic_autocast(self, device):
        if not hasattr(self.autocast_lists, "torch_16"):
            self.skipTest("No torch_16 list")
        for op_with_args in self.autocast_lists.torch_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            with torch.amp.autocast(device_type=self.device_type):
                output1 = getattr(torch, op)(*args, **maybe_kwargs)
            with torch.amp.autocast(device_type=self.device_type):
                output2 = getattr(torch, op)(*args, **maybe_kwargs)
            self.assertEqual(output1, output2)

    def test_cpu_autocast_deprecated_warning(self, device):
        if self.device_type != "cpu":
            self.skipTest("CPU-specific deprecated API test")
        with self.assertWarnsRegex(
            FutureWarning,
            r"`torch.cpu.amp.autocast\(args...\)` is deprecated. Please use `torch.amp.autocast\('cpu', args...\)` instead.",
        ):
            with torch.cpu.amp.autocast():
                _ = torch.ones(10)


instantiate_device_type_tests(
    TestAutocastOps, globals(), allow_mps=True, allow_xpu=True
)


class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_t):
        ctx.save_for_backward(x, w_t)
        ctx.device_type = x.device.type
        return torch.nn.functional.linear(x, w_t)

    @staticmethod
    def backward(ctx, grad_output):
        x, w_t = ctx.saved_tensors
        with torch.autocast(device_type=ctx.device_type):
            dL_dX = torch.matmul(grad_output, w_t)
            dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
        return dL_dX, dL_dW


class WeightDTypeCastCounterMode(TorchDispatchMode):
    def __init__(self, weight, target_dtype=torch.float16):
        super().__init__()
        self.dtype_cast_counter = 0
        self.weight = weight
        self.target_dtype = target_dtype

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            func is torch.ops.aten._to_copy.default
            and args[0] is self.weight
            and kwargs["dtype"] is self.target_dtype
        ):
            self.dtype_cast_counter += 1
        return func(*args, **kwargs)

    def __enter__(self):
        self.old_clear_cache = torch.clear_autocast_cache
        torch.clear_autocast_cache = lambda: None
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.clear_autocast_cache = self.old_clear_cache
        return super().__exit__(exc_type, exc_val, exc_tb)


class TestAutocastDeviceType(DeviceTypeTestBase):
    """Device-generic autocast tests that run across all device types."""

    def _get_amp_dtype(self):
        return torch.get_autocast_dtype(device_type=self.device_type)

    @onlyAccelerator
    @skipMPS  # MPS doesn't support cache-clearing mock; see TestAutocastMPS
    def test_cast_cache_is_global(self, device):
        """
        Verifies that the autocast cache is global. This is done by
        mocking out cache clearing at the end of the forward pass,
        running forward+backward with an explicit call to autocast in the
        backward, and verifying that the weight only gets cast once.
        """
        amp_dtype = self._get_amp_dtype()
        data = torch.randn(2, 3, device=device)
        weight = torch.nn.Parameter(torch.randn(4, 3, device=device))

        with WeightDTypeCastCounterMode(weight, target_dtype=amp_dtype) as mode:
            with torch.autocast(device_type=self.device_type):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()

        self.assertEqual(mode.dtype_cast_counter, 1)

    @onlyAccelerator
    @skipMPS  # MPS doesn't support cache-clearing mock; see TestAutocastMPS
    def test_cache_disabled(self, device):
        """
        Verifies that when a weight is marked as a cached tensor,
        autocast does NOT cache its conversion (cast happens twice).
        """
        amp_dtype = self._get_amp_dtype()
        data = torch.randn(2, 3, device=device)
        weight = torch.nn.Parameter(torch.randn(4, 3, device=device))

        try:
            torch._C._set_cached_tensors_enabled(True)
            torch._C._add_cached_tensor(weight)

            with WeightDTypeCastCounterMode(weight, target_dtype=amp_dtype) as mode:
                with torch.autocast(device_type=self.device_type):
                    output = CustomLinear.apply(data, weight)
                    s = output.sum()
                s.backward()

            self.assertEqual(mode.dtype_cast_counter, 2)

        finally:
            torch._C._set_cached_tensors_enabled(False)

    @onlyAccelerator
    def test_autocast_prioritize(self, device):
        """
        Test that index_put 'promote' policy works correctly under autocast.
        See https://github.com/pytorch/pytorch/issues/132715.
        """
        dtype = self._get_amp_dtype()

        with torch.autocast(device_type=self.device_type, enabled=True, dtype=dtype):
            t = torch.randn(
                [3, 4, 5], dtype=dtype, device=device, requires_grad=True
            )
            index = torch.randint(
                low=0, high=3, size=[3, 4, 5], dtype=torch.int64, device=device
            )
            val = torch.randn(1, dtype=dtype, device=device)

            res = torch.index_put(t, [index], val)

            loss = res.mean()
            loss.backward()

    def test_autocast_fast_dtype(self, device):
        """Verify the device reports a valid autocast fast dtype."""
        fast_dtype = torch.get_autocast_dtype(device_type=self.device_type)
        self.assertIn(fast_dtype, [torch.float16, torch.bfloat16])

    def test_autocast_nograd_caching_issue_158232(self, device):
        """
        Regression test for issue #158232: autocast + no_grad incompatibility.
        Ensures that exiting no_grad inside autocast restores gradient tracking.
        """
        dtype = self._get_amp_dtype()
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        with torch.autocast(self.device_type, dtype=dtype, enabled=True):
            with torch.no_grad():
                out1 = model(inp)
                self.assertFalse(
                    out1.requires_grad, "Output in no_grad should not require grad"
                )

            out2 = model(inp)
            self.assertTrue(
                out2.requires_grad,
                "Output should require gradients after exiting no_grad",
            )
            self.assertIsNotNone(
                out2.grad_fn, "Output should have grad_fn after exiting no_grad"
            )

            loss = out2.mean()
            loss.backward()

        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(model.bias.grad)

    def test_autocast_inference_mode_interaction(self, device):
        """
        Test that autocast works correctly with torch.inference_mode().
        """
        dtype = self._get_amp_dtype()
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        # Test 1: inference_mode inside autocast
        with torch.autocast(self.device_type, dtype=dtype, enabled=True):
            torch.clear_autocast_cache()
            with torch.inference_mode():
                out1 = model(inp)
                self.assertFalse(out1.requires_grad)
                self.assertEqual(out1.dtype, dtype)

            # After exiting inference_mode, gradients should work
            out2 = model(inp)
            self.assertTrue(out2.requires_grad)
            out2.mean().backward()

        # Test 2: autocast inside inference_mode
        with torch.inference_mode():
            with torch.autocast(self.device_type, dtype=dtype, enabled=True):
                out = model(inp)
                self.assertFalse(out.requires_grad)
                self.assertEqual(out.dtype, dtype)

    def test_autocast_caching_still_works_with_gradients(self, device):
        """
        Verify that autocast caching functions correctly when gradients ARE enabled.
        """
        dtype = self._get_amp_dtype()
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        with torch.autocast(self.device_type, dtype=dtype, enabled=True):
            out1 = model(inp)
            out2 = model(inp)
            out3 = model(inp)

            self.assertTrue(out1.requires_grad)
            self.assertTrue(out2.requires_grad)
            self.assertTrue(out3.requires_grad)

            self.assertIsNotNone(out1.grad_fn)
            self.assertIsNotNone(out2.grad_fn)
            self.assertIsNotNone(out3.grad_fn)

            out1.mean().backward(retain_graph=True)
            out2.mean().backward(retain_graph=True)
            out3.mean().backward()

    def test_autocast_mixed_grad_contexts(self, device):
        """
        Test complex nesting of gradient contexts within autocast.
        """
        dtype = self._get_amp_dtype()
        model = torch.nn.Linear(2, 2).to(device)
        inp = torch.randn(8, 2, device=device)

        with torch.autocast(self.device_type, dtype=dtype, enabled=True):
            with torch.no_grad():
                out1 = model(inp)
                self.assertFalse(out1.requires_grad)

            out2 = model(inp)
            self.assertTrue(out2.requires_grad)

            with torch.no_grad():
                out3 = model(inp)
                self.assertFalse(out3.requires_grad)

            out4 = model(inp)
            self.assertTrue(out4.requires_grad)

            (out2.mean() + out4.mean()).backward()


instantiate_device_type_tests(
    TestAutocastDeviceType, globals(), allow_mps=True, allow_xpu=True
)


@unittest.skipIf(not torch.backends.mps.is_available(), "requires mps")
class TestAutocastMPS(TestCase):
    def test_cast_cache_is_global(self):
        class CustomLinearMPS(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w_t):
                ctx.save_for_backward(x, w_t)
                return torch.nn.functional.linear(x, w_t)

            @staticmethod
            def backward(ctx, grad_output):
                x, w_t = ctx.saved_tensors
                with torch.autocast(device_type="mps"):
                    dL_dX = torch.matmul(grad_output, w_t)
                    dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(
                        0, 1
                    )
                return dL_dX, dL_dW

        data = torch.randn(2, 3).to("mps")
        weight = torch.nn.Parameter(torch.randn(4, 3).to("mps"))
        weight_dtype_cast_counter = 0

        class WeightDTypeCastCounterModeMPS(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if (
                    func is torch.ops.aten._to_copy.default
                    and args[0] is weight
                    and kwargs["dtype"] is torch.float16
                ):
                    nonlocal weight_dtype_cast_counter
                    weight_dtype_cast_counter += 1
                return func(*args, **kwargs)

            def __enter__(self):
                return super().__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                return super().__exit__(exc_type, exc_val, exc_tb)

        with WeightDTypeCastCounterModeMPS():
            with torch.autocast(device_type="mps"):
                output = CustomLinearMPS.apply(data, weight)
                s = output.sum()
            s.backward()
        self.assertEqual(weight_dtype_cast_counter, 2)

    def test_mps_autocast_error_message(self):
        with self.assertWarnsRegex(
            UserWarning,
            "MPS Autocast only supports dtypes of torch.bfloat16, torch.float16 currently.",
        ):
            with torch.autocast(device_type="mps", dtype=torch.float32):
                _ = torch.ones(10)

    def test_mps_autocast_bfloat16_supported(self):
        with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
            x = torch.randn(2, 3, device="mps")
            y = torch.randn(3, 3, device="mps")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.bfloat16)


class TestTorchAutocast(TestCase):
    """Tests for autocast API behavior that don't depend on any accelerator."""

    def test_invalid_device(self):
        dev = "not a real device"
        msg = f"Invalid device string: '{dev}'"
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.autocast(device_type=dev):
                _ = torch.tensor(1)
        with self.assertRaisesRegex(RuntimeError, msg):
            if not torch.amp.is_autocast_available(device_type=dev):
                raise AssertionError(f"autocast should be available for {dev}")

    def test_non_string_device(self):
        """Test that `autocast` throws a ValueError when provided a `torch.device` object for `device_type` instead of a string"""
        dev = torch.device("cpu")
        msg = f"Expected `device_type` of type `str`, got: `{type(dev)}`"
        with self.assertRaisesRegex(expected_exception=ValueError, expected_regex=msg):
            torch.autocast(device_type=dev)

    def test_autocast_called_with_non_callable(self):
        """Test that autocast gives a clear error when misused as a function wrapper"""
        x = torch.randn(2, 3)
        msg = r"autocast\(\)\(func\) requires a callable, but got Tensor"
        with self.assertRaisesRegex(TypeError, msg):
            torch.autocast(device_type="cpu")(x)


if __name__ == "__main__":
    run_tests()
