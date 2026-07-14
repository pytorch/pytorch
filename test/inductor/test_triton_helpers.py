# Owner(s): ["module: inductor"]

"""Tests for exclusive_scan_decoupled_lookback_64 dtype fix.

The fix ensures that `test_target` maintains consistent dtype in the
`if flag == 2` branch by using `tl.full([], -1, index.dtype)` instead
of the literal `-1` which would be int32.

This is a compile-time check - Triton validates type consistency across
all branches during compilation, so the test will fail to compile if
the dtype mismatch exists.
"""

import torch
from torch._inductor.runtime.triton_helpers import exclusive_scan_decoupled_lookback_64
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


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
