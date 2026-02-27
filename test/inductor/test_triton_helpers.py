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

import torch
from torch._inductor.runtime.triton_helpers import (
    exclusive_scan_decoupled_lookback_64,
    select_one,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


if HAS_GPU:
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


class ExclusiveScanDecoupledLookback64Test(TestCase):
    """Test cases for exclusive_scan_decoupled_lookback_64 dtype fix."""

    @requires_gpu()
    def test_flag_2_branch_with_int64_index(self) -> None:
        """Test `if flag == 2` branch with int64 index."""
        device = torch.device(GPU_TYPE)

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

    @requires_gpu()
    def test_flag_2_branch_with_int32_index(self) -> None:
        """Test `if flag == 2` branch with int32 index."""
        device = torch.device(GPU_TYPE)

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


class SelectOneTest(TestCase):
    """Test cases for select_one bitcast fix with sub-32-bit dtypes.

    The fix (D93872067) adds an intermediate .to(idtype) truncation before
    the final bitcast in select_one. Without this fix, tl.sum() promotes
    sub-32-bit unsigned integers (e.g. uint16) to int32, and the subsequent
    bitcast from int32 to a 16-bit dtype fails with a size mismatch error.
    """

    def _run_select_one(self, dtype: torch.dtype) -> None:
        device = torch.device(GPU_TYPE)
        BLOCK_SIZE = 4

        # Create input tensor and a one-hot mask selecting the element at index 2
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device)
        mask = torch.tensor([0, 0, 1, 0], dtype=torch.int32, device=device)
        result = torch.zeros(1, dtype=dtype, device=device)

        test_kernel_select_one[(1,)](x, mask, result, BLOCK_SIZE=BLOCK_SIZE)

        expected = torch.tensor([3.0], dtype=dtype, device=device)
        torch.testing.assert_close(result, expected)

    @requires_gpu()
    def test_select_one_bfloat16(self) -> None:
        """Test select_one with bfloat16 (16-bit) — triggers the bitcast fix."""
        self._run_select_one(torch.bfloat16)

    @requires_gpu()
    def test_select_one_float16(self) -> None:
        """Test select_one with float16 (16-bit) — triggers the bitcast fix."""
        self._run_select_one(torch.float16)

    @requires_gpu()
    def test_select_one_float32(self) -> None:
        """Test select_one with float32 (32-bit) — baseline that always worked."""
        self._run_select_one(torch.float32)

    @requires_gpu()
    def test_select_one_float64(self) -> None:
        """Test select_one with float64 (64-bit) — baseline that always worked."""
        self._run_select_one(torch.float64)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
