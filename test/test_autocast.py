# Owner(s): ["module: unknown"]

import unittest

import torch
from torch.testing._internal.autocast_test_lists import (
    AutocastCPUTestLists,
    TestAutocast,
)
from torch.testing._internal.common_device_type import expectedFailureMPSPre14
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


class TestAutocastCPU(TestAutocast):
    def setUp(self):
        super().setUp()
        self.autocast_lists = AutocastCPUTestLists(torch.device("cpu"))

    def tearDown(self):
        del self.autocast_lists
        super().tearDown()

    @skipIfTorchDynamo()
    def test_autocast_torch_expect_builtin_promote(self):
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(
                op, args1, torch.float32, device="cpu", out_type=out_type
            )
            self._run_autocast_outofplace(
                op,
                args2,
                torch.float32,
                device="cpu",
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_methods_expect_builtin_promote(self):
        for (
            op,
            args1,
            args2,
            out_type,
        ) in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(
                op, args1, torch.float32, device="cpu", module=None, out_type=out_type
            )
            self._run_autocast_outofplace(
                op,
                args2,
                torch.float32,
                device="cpu",
                module=None,
                out_type=out_type,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_16(self):
        for op_with_args in self.autocast_lists.torch_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, torch.bfloat16, device="cpu", add_kwargs=maybe_kwargs
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                device="cpu",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_nn_16(self):
        for op_with_args in self.autocast_lists.nn_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                torch.bfloat16,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float16,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_fp32(self):
        for op_with_args in self.autocast_lists.torch_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op, args, torch.float32, device="cpu", add_kwargs=maybe_kwargs
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cpu",
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_nn_fp32(self):
        for op_with_args in self.autocast_lists.nn_fp32:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
            )
            self._run_autocast_outofplace(
                op,
                args,
                torch.float32,
                device="cpu",
                module=torch._C._nn,
                add_kwargs=maybe_kwargs,
                amp_dtype=torch.float16,
            )

    @skipIfTorchDynamo()
    def test_autocast_torch_need_autocast_promote(self):
        for op, args1, args2 in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args1, torch.float32, device="cpu")
            self._run_autocast_outofplace(
                op, args2, torch.float32, device="cpu", amp_dtype=torch.float16
            )

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def test_autocast_rnn(self):
        if (
            torch.backends.onednn.is_available()
            and torch.ops.onednn._is_onednn_bf16_supported()
        ):
            x = torch.randn(1, 2, 1)
            hx = torch.randn(2, 2, 1)
            cx = torch.randn(2, 2, 1)

            m = torch.nn.LSTM(1, 1, 2).to(torch.bfloat16)

            # Raise ValueError when autocast is not enabled
            with self.assertRaisesRegex(ValueError, "input must have the type"):
                m(x, (hx, cx))

            # Should be able to run the below case with autocast
            with torch.amp.autocast(device_type="cpu"):
                m(x, (hx, cx))

    def test_autocast_disabled_with_fp32_dtype(self):
        with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False):
            _ = torch.ones(10)

    def test_generic_autocast(self):
        for op_with_args in self.autocast_lists.torch_16:
            op, args, maybe_kwargs = self.args_maybe_kwargs(op_with_args)
            with torch.amp.autocast(device_type="cpu"):
                generic_autocast_output = getattr(torch, op)(*args, **maybe_kwargs)
            with torch.amp.autocast(device_type="cpu"):
                cpu_autocast_output = getattr(torch, op)(*args, **maybe_kwargs)
            self.assertEqual(generic_autocast_output, cpu_autocast_output)

    def test_cpu_autocast_deprecated_warning(self):
        with self.assertWarnsRegex(
            FutureWarning,
            r"`torch.cpu.amp.autocast\(args...\)` is deprecated. Please use `torch.amp.autocast\('cpu', args...\)` instead.",
        ):
            with torch.cpu.amp.autocast():
                _ = torch.ones(10)


class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_t):
        ctx.save_for_backward(x, w_t)
        return torch.nn.functional.linear(x, w_t)

    @staticmethod
    def backward(ctx, grad_output):
        x, w_t = ctx.saved_tensors
        with torch.autocast(device_type="cuda"):
            dL_dX = torch.matmul(grad_output, w_t)
            dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
        return dL_dX, dL_dW


class WeightDTypeCastCounterMode(TorchDispatchMode):
    def __init__(self, weight):
        super().__init__()
        self.dtype_cast_counter = 0
        self.weight = weight

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            func is torch.ops.aten._to_copy.default
            and args[0] is self.weight
            and kwargs["dtype"] is torch.float16
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


@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
class TestAutocastGPU(TestCase):
    def test_cast_cache_is_global(self):
        """
        Verifies that the autocast cache is global. This is done by
        mocking out cache clearing at the end of the forward pass,
        running forward+backward with an explicit call to autocast in the
        backward, and verifying that the weight only get cast to float16 once.
        """

        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())

        with WeightDTypeCastCounterMode(weight) as mode:
            with torch.autocast(device_type="cuda"):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()

        self.assertEqual(mode.dtype_cast_counter, 1)

    def test_cache_disabled(self):
        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())

        try:
            torch._C._set_cached_tensors_enabled(True)
            torch._C._add_cached_tensor(weight)

            with WeightDTypeCastCounterMode(weight) as mode:
                with torch.autocast(device_type="cuda"):
                    output = CustomLinear.apply(data, weight)
                    s = output.sum()
                s.backward()

            # we should not have cached the conversion of the weight
            self.assertEqual(mode.dtype_cast_counter, 2)

        finally:
            torch._C._set_cached_tensors_enabled(False)

    # index_put under AMP follows a cast policy called "promote",
    # https://github.com/pytorch/pytorch/blob/4fcd15a667df5b80e81db6563d8d3123a0cbd051/aten/src/ATen/autocast_mode.h#L205-L230
    # That means:
    #   (1) double precision is ignored,
    #   (2) if any argument is float, then all arguments are promoted to float,
    #   (3) if all arguments are of lower precision dtype, then all dtypes must be equal to the same amp autocast dtype.
    # Since AMP autocast dtype is thread-local, it is not preserved across thread boundaries during autograd execution,
    # and due to the multi-threaded nature of the autograd, the forward pass is being run in bfloat16, while the backward
    # pass defaults to float16. The dtype mismatch leads to the error in the policy, as the criteria (3) is not satisfied.
    # For more info see https://github.com/pytorch/pytorch/issues/132715.
    def test_autocast_prioritize(self):
        device = "cuda"
        dtype = torch.bfloat16

        with torch.autocast(device_type=device, enabled=True, dtype=dtype):
            t = torch.randn([3, 4, 5], dtype=dtype, device=device, requires_grad=True)
            index = torch.randint(
                low=0, high=3, size=[3, 4, 5], dtype=torch.int64, device=device
            )
            val = torch.randn(1, dtype=dtype, device=device)

            res = torch.index_put(t, [index], val)

            loss = res.mean()
            loss.backward()


@unittest.skipIf(not torch.backends.mps.is_available(), "requires mps")
class TestAutocastMPS(TestCase):
    def test_cast_cache_is_global(self):
        class CustomLinear(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w_t):
                ctx.save_for_backward(x, w_t)
                return torch.nn.functional.linear(x, w_t)

            @staticmethod
            def backward(ctx, grad_output):
                x, w_t = ctx.saved_tensors
                with torch.autocast(device_type="mps"):
                    dL_dX = torch.matmul(grad_output, w_t)
                    dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
                return dL_dX, dL_dW

        data = torch.randn(2, 3).to("mps")
        weight = torch.nn.Parameter(torch.randn(4, 3).to("mps"))
        weight_dtype_cast_counter = 0

        class WeightDTypeCastCounterMode(TorchDispatchMode):
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
                # self.old_clear_cache = torch.clear_autocast_cache
                # torch.clear_autocast_cache = lambda: None
                return super().__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                # torch.clear_autocast_cache = self.old_clear_cache
                return super().__exit__(exc_type, exc_val, exc_tb)

        with WeightDTypeCastCounterMode():
            with torch.autocast(device_type="mps"):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()
        self.assertEqual(weight_dtype_cast_counter, 2)

    def test_mps_autocast_error_message(self):
        with self.assertWarnsRegex(
            UserWarning,
            "MPS Autocast only supports dtype of torch.bfloat16 and torch.float16 currently.",
        ):
            with torch.autocast(device_type="mps", dtype=torch.float32):
                _ = torch.ones(10)

    # torch.bfloat16 is only supported on macOS 14 and above.
    @expectedFailureMPSPre14
    def test_mps_autocast_bfloat16_supported(self):
        with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16):
            x = torch.randn(2, 3, device="mps")
            y = torch.randn(3, 3, device="mps")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.bfloat16)


class TestTorchAutocast(TestCase):
    def test_autocast_fast_dtype(self):
        gpu_fast_dtype = torch.get_autocast_dtype(device_type="cuda")
        cpu_fast_dtype = torch.get_autocast_dtype(device_type="cpu")
        self.assertEqual(gpu_fast_dtype, torch.half)
        self.assertEqual(cpu_fast_dtype, torch.bfloat16)

    def test_invalid_device(self):
        dev = "not a real device"
        msg = f"Invalid device string: '{dev}'"
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.autocast(device_type=dev):
                _ = torch.tensor(1)
        with self.assertRaisesRegex(RuntimeError, msg):
            assert torch.amp.is_autocast_available(device_type=dev)

    def test_non_string_device(self):
        """Test that `autocast` throws a ValueError when provided a `torch.device` object for `device_type` instead of a string"""
        dev = torch.device("cpu")
        msg = f"Expected `device_type` of type `str`, got: `{type(dev)}`"
        with self.assertRaisesRegex(expected_exception=ValueError, expected_regex=msg):
            torch.autocast(device_type=dev)


if __name__ == "__main__":
    run_tests()
