# Owner(s): ["module: inductor"]

"""Tests for triton_helpers functions.

Covers:
- exclusive_scan_decoupled_lookback_64 dtype fix (D89705211): ensures
  `test_target` maintains consistent dtype by using `tl.full([], -1, index.dtype)`
  instead of the literal `-1`.
- select_one bitcast fix for sub-32-bit dtypes (D93872067): ensures the
  intermediate result from `tl.sum()` is truncated back to the original-width
  integer type before the final bitcast, preventing size mismatch errors.
"""

import unittest

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.exc import TritonUnavailableError
from torch._inductor.runtime.triton_helpers import (
    exclusive_scan_decoupled_lookback_64,
    max2,
    max2_strict,
    maximum,
    min2,
    min2_strict,
    minimum,
    rand4x,
    randn4x,
    select_one,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import HardwareClassification
from torch.testing._internal.inductor_utils import HAS_TRITON, requires_triton


if HAS_TRITON:
    import triton  # @manual
    from triton import language as tl

    @triton.jit
    def _add_combine_fn(a, b):
        return a + b

    @triton.jit
    def test_kernel_exclusive_scan(
        scratch_ptr,
        block_value_ptr,
        index_ptr,
        result_ptr,
    ):
        block_value = tl.load(block_value_ptr)
        index = tl.load(index_ptr)

        exclusive_prefix = exclusive_scan_decoupled_lookback_64(
            scratch_ptr,
            block_value,
            index,
            _add_combine_fn,
        )

        tl.store(result_ptr, exclusive_prefix)

    @triton.jit
    def test_kernel_select_one(
        x_ptr,
        mask_ptr,
        result_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        mask = tl.load(mask_ptr + offsets)
        result = select_one(x, mask, dim=0)
        tl.store(result_ptr, result)

    @triton.jit
    def test_kernel_random_4x_order(
        seed,
        helper_result_ptr,
        expected_result_ptr,
        BLOCK_SIZE: tl.constexpr,
        NORMAL: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)

        if BLOCK_SIZE >= 4 and BLOCK_SIZE % 4 == 0:
            quarter_block_size: tl.constexpr = BLOCK_SIZE // 4
            reduced_offsets = tl.arange(0, quarter_block_size)

            if NORMAL:
                helper = randn4x(seed, offsets, BLOCK_SIZE)
                r0, r1, r2, r3 = tl.randn4x(seed, reduced_offsets)
            else:
                helper = rand4x(seed, offsets, BLOCK_SIZE)
                r0, r1, r2, r3 = tl.rand4x(seed, reduced_offsets)

            tl.store(expected_result_ptr + 4 * reduced_offsets, r0)
            tl.store(expected_result_ptr + 4 * reduced_offsets + 1, r1)
            tl.store(expected_result_ptr + 4 * reduced_offsets + 2, r2)
            tl.store(expected_result_ptr + 4 * reduced_offsets + 3, r3)
        else:
            if NORMAL:
                helper = randn4x(seed, offsets, BLOCK_SIZE)
                expected = tl.randn(seed, offsets)
            else:
                helper = rand4x(seed, offsets, BLOCK_SIZE)
                expected = tl.rand(seed, offsets)

            tl.store(expected_result_ptr + offsets, expected)

        tl.store(helper_result_ptr + offsets, helper)

    @triton.jit
    def test_kernel_random_4x_distribution(
        seed,
        result_ptr,
        BLOCK_SIZE: tl.constexpr,
        NORMAL: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if NORMAL:
            result = randn4x(seed, offsets, BLOCK_SIZE)
        else:
            result = rand4x(seed, offsets, BLOCK_SIZE)
        tl.store(result_ptr + offsets, result)

    @triton.jit
    def test_kernel_minimum_maximum(
        a_ptr,
        b_ptr,
        min_ptr,
        max_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        a = tl.load(a_ptr + offsets)
        b = tl.load(b_ptr + offsets)
        tl.store(min_ptr + offsets, minimum(a, b))
        tl.store(max_ptr + offsets, maximum(a, b))

    @triton.jit
    def test_kernel_minmax_reduction(
        x_ptr,
        min_ptr,
        max_ptr,
        BLOCK_SIZE: tl.constexpr,
        STRICT: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        if STRICT:
            tl.store(min_ptr + row, min2_strict(x, 0))
            tl.store(max_ptr + row, max2_strict(x, 0))
        else:
            tl.store(min_ptr + row, min2(x, 0))
            tl.store(max_ptr + row, max2(x, 0))


class _TritonDeviceTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        device = cls.get_primary_device()
        if not HAS_TRITON:
            raise unittest.SkipTest(f"triton is required for {device}")
        try:
            device_interface = get_interface_for_device(torch.device(device).type)
        except NotImplementedError as exc:
            raise unittest.SkipTest(f"requires Triton support for {device}") from exc
        if not device_interface.is_triton_capable(device):
            raise unittest.SkipTest(f"requires Triton support for {device}")
        try:
            device_interface.raise_if_triton_unavailable(device)
        except TritonUnavailableError as exc:
            raise unittest.SkipTest(str(exc)) from exc


class ExclusiveScanDecoupledLookback64Test(_TritonDeviceTestCase):
    """Test cases for exclusive_scan_decoupled_lookback_64 dtype fix."""

    hw_classification = HardwareClassification.ACCELERATOR

    @onlyAccelerator
    @requires_triton()
    def test_flag_2_branch_with_int64_index(self, device) -> None:
        """Test `if flag == 2` branch with int64 index."""

        # Scratch memory layout per block: [flag, partial_aggregate, inclusive_prefix]
        # Block 0: flag=2 (inclusive prefix ready), inclusive_prefix=10.0
        scratch = torch.zeros(6, dtype=torch.uint64, device=device)
        scratch[0] = 2
        inclusive_prefix_value = torch.tensor(
            [10.0], dtype=torch.float64, device=device
        )
        scratch[2] = inclusive_prefix_value.view(torch.int64).item()

        block_value = torch.tensor([5.0], dtype=torch.float64, device=device)
        index = torch.tensor([1], dtype=torch.int64, device=device)
        result = torch.zeros(1, dtype=torch.float64, device=device)

        test_kernel_exclusive_scan[(1,)](scratch, block_value, index, result)

        # Block 1's exclusive prefix = Block 0's inclusive prefix = 10.0
        expected = torch.tensor([10.0], dtype=torch.float64, device=device)
        torch.testing.assert_close(result, expected)

    @onlyAccelerator
    @requires_triton()
    def test_flag_2_branch_with_int32_index(self, device) -> None:
        """Test `if flag == 2` branch with int32 index."""

        # Scratch memory layout per block: [flag, partial_aggregate, inclusive_prefix]
        # Block 0: flag=2 (inclusive prefix ready), inclusive_prefix=10.0
        scratch = torch.zeros(6, dtype=torch.uint64, device=device)
        scratch[0] = 2
        inclusive_prefix_value = torch.tensor(
            [10.0], dtype=torch.float64, device=device
        )
        scratch[2] = inclusive_prefix_value.view(torch.int64).item()

        block_value = torch.tensor([5.0], dtype=torch.float64, device=device)
        index = torch.tensor([1], dtype=torch.int32, device=device)
        result = torch.zeros(1, dtype=torch.float64, device=device)

        test_kernel_exclusive_scan[(1,)](scratch, block_value, index, result)

        # Block 1's exclusive prefix = Block 0's inclusive prefix = 10.0
        expected = torch.tensor([10.0], dtype=torch.float64, device=device)
        torch.testing.assert_close(result, expected)


class SelectOneTest(_TritonDeviceTestCase):
    """Test cases for select_one bitcast fix with sub-32-bit dtypes.

    The fix (D93872067) adds an intermediate .to(idtype) truncation before
    the final bitcast in select_one. Without this fix, tl.sum() promotes
    sub-32-bit unsigned integers (e.g. uint16) to int32, and the subsequent
    bitcast from int32 to a 16-bit dtype fails with a size mismatch error.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def _run_select_one(self, device, dtype: torch.dtype) -> None:
        BLOCK_SIZE = 4

        # Create input tensor and a one-hot mask selecting the element at index 2
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device)
        mask = torch.tensor([0, 0, 1, 0], dtype=torch.int32, device=device)
        result = torch.zeros(1, dtype=dtype, device=device)

        test_kernel_select_one[(1,)](x, mask, result, BLOCK_SIZE=BLOCK_SIZE)

        expected = torch.tensor([3.0], dtype=dtype, device=device)
        torch.testing.assert_close(result, expected)

    @onlyAccelerator
    @requires_triton()
    def test_select_one_bfloat16(self, device) -> None:
        """Test select_one with bfloat16 (16-bit) — triggers the bitcast fix."""
        self._run_select_one(device, torch.bfloat16)

    @onlyAccelerator
    @requires_triton()
    def test_select_one_float16(self, device) -> None:
        """Test select_one with float16 (16-bit) — triggers the bitcast fix."""
        self._run_select_one(device, torch.float16)

    @onlyAccelerator
    @requires_triton()
    def test_select_one_float32(self, device) -> None:
        """Test select_one with float32 (32-bit) — baseline that always worked."""
        self._run_select_one(device, torch.float32)

    @onlyAccelerator
    @requires_triton()
    def test_select_one_float64(self, device) -> None:
        """Test select_one with float64 (64-bit) — baseline that always worked."""
        self._run_select_one(device, torch.float64)


class Random4xTest(_TritonDeviceTestCase):
    """Test cases for rand4x/randn4x helper packing order."""

    hw_classification = HardwareClassification.ACCELERATOR

    def _run_random_4x_order(self, device, normal: bool, block_size: int) -> None:
        helper_result = torch.empty(block_size, dtype=torch.float32, device=device)
        expected_result = torch.empty(block_size, dtype=torch.float32, device=device)

        test_kernel_random_4x_order[(1,)](
            1234,
            helper_result,
            expected_result,
            BLOCK_SIZE=block_size,
            NORMAL=normal,
        )

        torch.testing.assert_close(helper_result, expected_result, atol=0, rtol=0)

    @onlyAccelerator
    @requires_triton()
    def test_rand4x_order(self, device) -> None:
        self._run_random_4x_order(device, normal=False, block_size=16)

    @onlyAccelerator
    @requires_triton()
    def test_randn4x_order(self, device) -> None:
        self._run_random_4x_order(device, normal=True, block_size=16)

    @onlyAccelerator
    @requires_triton()
    def test_rand4x_order_quarter_block_size_2(self, device) -> None:
        self._run_random_4x_order(device, normal=False, block_size=8)

    @onlyAccelerator
    @requires_triton()
    def test_randn4x_order_quarter_block_size_2(self, device) -> None:
        self._run_random_4x_order(device, normal=True, block_size=8)

    @onlyAccelerator
    @requires_triton()
    def test_rand4x_fallback_block_size_2(self, device) -> None:
        self._run_random_4x_order(device, normal=False, block_size=2)

    @onlyAccelerator
    @requires_triton()
    def test_randn4x_fallback_block_size_2(self, device) -> None:
        self._run_random_4x_order(device, normal=True, block_size=2)

    def _run_random_4x_block_size_stability(self, device, normal: bool) -> None:
        sample_count = 1024
        small_block_result = torch.empty(
            sample_count, dtype=torch.float32, device=device
        )
        large_block_result = torch.empty(
            sample_count, dtype=torch.float32, device=device
        )

        test_kernel_random_4x_distribution[(sample_count // 8,)](
            1234,
            small_block_result,
            BLOCK_SIZE=8,
            NORMAL=normal,
        )
        test_kernel_random_4x_distribution[(sample_count // 1024,)](
            1234,
            large_block_result,
            BLOCK_SIZE=1024,
            NORMAL=normal,
        )

        torch.testing.assert_close(small_block_result, large_block_result)

    @onlyAccelerator
    @requires_triton()
    def test_rand4x_block_size_stability(self, device) -> None:
        self._run_random_4x_block_size_stability(device, normal=False)

    @onlyAccelerator
    @requires_triton()
    def test_randn4x_block_size_stability(self, device) -> None:
        self._run_random_4x_block_size_stability(device, normal=True)

    @onlyAccelerator
    @requires_triton()
    def test_rand4x_distribution(self, device) -> None:
        block_size = 1024
        num_blocks = 128
        sample_count = block_size * num_blocks
        result = torch.empty(sample_count, dtype=torch.float32, device=device)

        test_kernel_random_4x_distribution[(num_blocks,)](
            1234,
            result,
            BLOCK_SIZE=block_size,
            NORMAL=False,
        )

        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLess(result.max().item(), 1.0)
        self.assertLess(abs(result.mean().item() - 0.5), 0.01)
        self.assertLess(abs(result.var(unbiased=False).item() - (1.0 / 12.0)), 0.01)

        bins = torch.histc(result, bins=10, min=0.0, max=1.0)
        max_bucket_error = (bins - sample_count / 10).abs().max().item()
        self.assertLess(max_bucket_error / (sample_count / 10), 0.08)

    @onlyAccelerator
    @requires_triton()
    def test_randn4x_distribution(self, device) -> None:
        block_size = 1024
        num_blocks = 128
        sample_count = block_size * num_blocks
        result = torch.empty(sample_count, dtype=torch.float32, device=device)

        test_kernel_random_4x_distribution[(num_blocks,)](
            1234,
            result,
            BLOCK_SIZE=block_size,
            NORMAL=True,
        )

        mean = result.mean().item()
        centered = result - mean
        variance = centered.square().mean().item()
        skewness = (centered.pow(3).mean() / (variance**1.5)).item()

        self.assertLess(abs(mean), 0.02)
        self.assertLess(abs(variance - 1.0), 0.05)
        self.assertLess(abs(skewness), 0.05)


class MinimumMaximumTest(_TritonDeviceTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_elementwise_nan_and_signed_zero(self, device: str) -> None:
        a = torch.tensor(
            [
                -0.0,
                0.0,
                float("nan"),
                1.0,
                float("nan"),
                float("inf"),
                -float("inf"),
                2.0,
            ],
            device=device,
        )
        b = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                float("nan"),
                float("nan"),
                -float("inf"),
                float("inf"),
                -2.0,
            ],
            device=device,
        )
        actual_min = torch.empty_like(a)
        actual_max = torch.empty_like(a)
        test_kernel_minimum_maximum[(1,)](
            a,
            b,
            actual_min,
            actual_max,
            BLOCK_SIZE=a.numel(),
        )

        expected_min = torch.minimum(a, b)
        expected_max = torch.maximum(a, b)
        self.assertEqual(actual_min, expected_min)
        self.assertEqual(actual_max, expected_max)
        self.assertEqual(
            actual_min[:2].view(torch.int32), expected_min[:2].view(torch.int32)
        )
        self.assertEqual(
            actual_max[:2].view(torch.int32), expected_max[:2].view(torch.int32)
        )

    def test_reduction_nan(self, device: str) -> None:
        x = torch.tensor(
            [
                [1.0, -2.0, 3.0, float("nan")],
                [float("inf"), 2.0, -float("inf"), 1.0],
            ],
            device=device,
        )
        actual_min = torch.empty(x.shape[0], device=device)
        actual_max = torch.empty(x.shape[0], device=device)
        test_kernel_minmax_reduction[(x.shape[0],)](
            x,
            actual_min,
            actual_max,
            BLOCK_SIZE=x.shape[1],
            STRICT=False,
        )

        self.assertEqual(actual_min, torch.amin(x, dim=1))
        self.assertEqual(actual_max, torch.amax(x, dim=1))

    def test_reduction_signed_zero(self, device: str) -> None:
        x = torch.tensor(
            [
                [-0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0, -0.0, -0.0],
                [-0.0, 0.0, -0.0, 0.0],
                [0.0, -0.0, 0.0, -0.0],
            ],
            device=device,
        )
        actual_min = torch.empty(x.shape[0], device=device)
        actual_max = torch.empty(x.shape[0], device=device)
        test_kernel_minmax_reduction[(x.shape[0],)](
            x,
            actual_min,
            actual_max,
            BLOCK_SIZE=x.shape[1],
            STRICT=True,
        )

        expected_min = torch.amin(x, dim=1)
        expected_max = torch.amax(x, dim=1)
        self.assertEqual(actual_min.view(torch.int32), expected_min.view(torch.int32))
        self.assertEqual(actual_max.view(torch.int32), expected_max.view(torch.int32))

    def test_reduction_relaxed_signed_zero(self, device: str) -> None:
        x = torch.tensor(
            [
                [-0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0, -0.0, -0.0],
            ],
            device=device,
        )
        actual_min = torch.empty(x.shape[0], device=device)
        actual_max = torch.empty(x.shape[0], device=device)
        test_kernel_minmax_reduction[(x.shape[0],)](
            x,
            actual_min,
            actual_max,
            BLOCK_SIZE=x.shape[1],
            STRICT=False,
        )

        actual_min_bits = actual_min.view(torch.int32)
        actual_max_bits = actual_max.view(torch.int32)
        self.assertEqual(actual_min_bits, torch.full_like(actual_min_bits, -(1 << 31)))
        self.assertEqual(actual_max_bits, torch.zeros_like(actual_max_bits))


instantiate_device_type_tests(
    ExclusiveScanDecoupledLookback64Test,
    globals(),
    except_for=("cpu", "hpu"),
    allow_xpu=True,
)
instantiate_device_type_tests(
    SelectOneTest,
    globals(),
    except_for=("cpu", "hpu"),
    allow_xpu=True,
)
instantiate_device_type_tests(
    Random4xTest,
    globals(),
    except_for=("cpu", "hpu"),
    allow_xpu=True,
)

if HAS_TRITON:
    instantiate_device_type_tests(
        MinimumMaximumTest,
        globals(),
        except_for=("cpu", "hpu"),
        allow_xpu=True,
    )


if __name__ == "__main__":
    if HAS_TRITON:
        run_tests()
