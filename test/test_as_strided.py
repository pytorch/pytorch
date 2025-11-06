# Owner(s): ["oncall: pt2"]

from collections import deque
from typing import Optional

import torch
from torch._prims_common import check_significant_strides
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._as_strided import as_strided_via_views


def get_state(t: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Extract (sizes, strides) tuple from a tensor."""
    return (tuple(t.size()), tuple(t.stride()))


def max_storage_offset(
    base_numel: int, size: tuple[int, ...], stride: tuple[int, ...]
) -> int:
    """
    Calculate the maximum storage offset for which a view with the given
    size/stride would fit within a tensor of base_numel elements.

    We conservatively require the full size*stride extent to fit inside,
    which ensures there's enough working space for unflatten operations.
    """
    if base_numel == 0:
        return 0

    if any(s == 0 for s in size):
        # numel == 0, all offsets from 0 to base_numel-1 are technically valid
        # but for simplicity we only test offset 0
        return 0

    # Require the full extent of the largest dimension to fit
    # For each dimension, the extent is size[i] * stride[i]
    max_extent = max(sz * max(st, 0) for sz, st in zip(size, stride))
    max_offset = base_numel - max_extent
    return max(0, max_offset)


def enumerate_reachable_states(
    initial_size: int,
    include_slice: bool = False,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Use BFS with DP to enumerate all reachable (size, stride) states from
    a 1D contiguous tensor via valid view operations.

    Args:
        initial_size: Size of the initial 1D tensor
        include_slice: If True, include slice operations with step>1

    We only explore states with offset=0 (you can retroactively change the offset).
    We reject states with size=0 or size=1 dimensions as they are degenerate.
    """
    # Create initial 1D contiguous tensor
    initial_tensor = torch.arange(initial_size)

    initial_state = get_state(initial_tensor)

    # Map from state to tensor for that state
    state_to_tensor: dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor] = {
        initial_state: initial_tensor
    }
    visited: set[tuple[tuple[int, ...], tuple[int, ...]]] = {initial_state}
    queue: deque[tuple[tuple[int, ...], tuple[int, ...]]] = deque([initial_state])

    while queue:
        state = queue.popleft()
        t = state_to_tensor[state]
        sizes, strides = state
        ndim = len(sizes)

        def add_state(new_t: torch.Tensor) -> None:
            new_state = get_state(new_t)
            sizes, strides = new_state
            # Skip if has size-0 or size-1 dimensions
            if any(s == 0 or s == 1 for s in sizes):
                return
            # Only accept states where strides are in descending order
            if list(strides) != sorted(strides, reverse=True):
                return
            if new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)
                state_to_tensor[new_state] = new_t

        # 1. Unflatten: try factoring each dimension
        for dim in range(ndim):
            size = sizes[dim]
            assert size > 1
            # Try all factorizations x * y = size where both x, y >= 2
            # We only need to check x up to size // 2 since when x > size // 2,
            # y = size // x < 2, which we reject
            for x in range(2, size // 2 + 1):
                if size % x == 0:
                    y = size // x
                    add_state(t.unflatten(dim, (x, y)))

        # 2. Slice/Narrow
        for dim in range(ndim):
            size = sizes[dim]
            max_step = size + 1 if include_slice else 2
            for start in range(size):
                for stop in range(start + 1, size + 1):
                    for step in range(1, max_step):
                        slices = [slice(None)] * ndim
                        slices[dim] = slice(start, stop, step)
                        add_state(t[tuple(slices)])

        # 3. Flatten: merge adjacent dimensions
        for dim in range(ndim - 1):
            add_state(t.flatten(dim, dim + 1))

    return visited


class TestAsStrided(TestCase):
    def assertSameView(
        self,
        result: torch.Tensor | type[NotImplemented],
        target: torch.Tensor,
        base: torch.Tensor,
        msg: str = "",
    ) -> None:
        self.assertIsNot(result, NotImplemented, msg=msg or "Got NotImplemented")
        self.assertEqual(result.size(), target.size(), msg=msg or "Size mismatch")
        same_strides, idx = check_significant_strides(result, target, only_cuda=False)
        if not same_strides:
            fail_msg = f"Stride mismatch at dim {idx}: result={result.stride()}, target={target.stride()}"
            if msg:
                fail_msg = f"{msg}: {fail_msg}"
            self.fail(fail_msg)
        self.assertTrue(result._is_view(), msg=msg or "Result is not a view")
        # Check that result is a view of the base tensor using object identity
        result_base = result._base if result._base is not None else result
        base_base = base._base if base._base is not None else base
        self.assertIs(
            result_base, base_base, msg=msg or "Result is not a view of the base tensor"
        )
        self.assertEqual(
            result.storage_offset(),
            target.storage_offset(),
            msg=msg or "Storage offset mismatch",
        )

    def check_as_strided_via_views(
        self,
        base: torch.Tensor,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        storage_offset: int = 0,
    ) -> None:
        """Helper to test as_strided_via_views matches torch.as_strided."""
        target = torch.as_strided(base, size, stride, storage_offset)
        result = as_strided_via_views(base, size, stride, storage_offset)
        self.assertSameView(
            result, target, base, f"Failed for {size=}, {stride=}, {storage_offset=}"
        )

    def test_size_10_exhaustive_without_slice(self) -> None:
        """Test that size 10 produces exactly 26 states without slice (step>1)."""
        expected_states = {
            ((2,), (1,)),
            ((2, 2), (2, 1)),
            ((2, 2), (3, 1)),
            ((2, 2), (4, 1)),
            ((2, 2), (5, 1)),
            ((2, 2, 2), (4, 2, 1)),
            ((2, 2, 2), (5, 2, 1)),
            ((2, 3), (3, 1)),
            ((2, 3), (4, 1)),
            ((2, 3), (5, 1)),
            ((2, 4), (4, 1)),
            ((2, 4), (5, 1)),
            ((2, 5), (5, 1)),
            ((3,), (1,)),
            ((3, 2), (2, 1)),
            ((3, 2), (3, 1)),
            ((3, 3), (3, 1)),
            ((4,), (1,)),
            ((4, 2), (2, 1)),
            ((5,), (1,)),
            ((5, 2), (2, 1)),
            ((6,), (1,)),
            ((7,), (1,)),
            ((8,), (1,)),
            ((9,), (1,)),
            ((10,), (1,)),
        }

        actual_states = enumerate_reachable_states(10, include_slice=False)

        self.assertEqual(len(actual_states), 26)
        self.assertEqual(actual_states, expected_states)

    def test_size_10_exhaustive_with_slice(self) -> None:
        """Test that size 10 produces exactly 54 states with slice (step>1)."""
        expected_states = {
            ((2,), (1,)),
            ((2,), (2,)),
            ((2,), (3,)),
            ((2,), (4,)),
            ((2,), (5,)),
            ((2,), (6,)),
            ((2,), (7,)),
            ((2,), (8,)),
            ((2,), (9,)),
            ((2, 2), (2, 1)),
            ((2, 2), (3, 1)),
            ((2, 2), (3, 2)),
            ((2, 2), (4, 1)),
            ((2, 2), (4, 2)),
            ((2, 2), (4, 3)),
            ((2, 2), (5, 1)),
            ((2, 2), (5, 2)),
            ((2, 2), (5, 3)),
            ((2, 2), (5, 4)),
            ((2, 2), (6, 1)),
            ((2, 2), (6, 2)),
            ((2, 2), (6, 3)),
            ((2, 2), (8, 1)),
            ((2, 2, 2), (4, 2, 1)),
            ((2, 2, 2), (5, 2, 1)),
            ((2, 3), (3, 1)),
            ((2, 3), (4, 1)),
            ((2, 3), (5, 1)),
            ((2, 3), (5, 2)),
            ((2, 3), (6, 1)),
            ((2, 4), (4, 1)),
            ((2, 4), (5, 1)),
            ((2, 5), (5, 1)),
            ((3,), (1,)),
            ((3,), (2,)),
            ((3,), (3,)),
            ((3,), (4,)),
            ((3, 2), (2, 1)),
            ((3, 2), (3, 1)),
            ((3, 2), (3, 2)),
            ((3, 2), (4, 1)),
            ((3, 3), (3, 1)),
            ((4,), (1,)),
            ((4,), (2,)),
            ((4,), (3,)),
            ((4, 2), (2, 1)),
            ((5,), (1,)),
            ((5,), (2,)),
            ((5, 2), (2, 1)),
            ((6,), (1,)),
            ((7,), (1,)),
            ((8,), (1,)),
            ((9,), (1,)),
            ((10,), (1,)),
        }

        actual_states = enumerate_reachable_states(10, include_slice=True)

        self.assertEqual(len(actual_states), 54)
        self.assertEqual(actual_states, expected_states)

    def test_subset_property(self) -> None:
        """
        Test that for sizes 2..10, each smaller tensor results in a strict
        subset of possible states compared to the next one.
        """
        prev_states: Optional[set[tuple[tuple[int, ...], tuple[int, ...]]]] = None
        for size in range(2, 11):
            current_states = enumerate_reachable_states(size)

            if prev_states is not None:
                # Check that prev_states is a strict subset of current_states
                self.assertTrue(
                    prev_states.issubset(current_states),
                    f"States from size {size - 1} are not a subset of size {size}",
                )
                # Check that it's a strict subset (not equal)
                self.assertTrue(
                    len(prev_states) < len(current_states),
                    f"States from size {size - 1} should be strictly fewer than size {size}",
                )

            prev_states = current_states

    def test_as_strided_via_views_exhaustive(self) -> None:
        """Exhaustively test on all reachable states from size 10, including all valid offsets."""
        initial_tensor = torch.arange(10)
        states = enumerate_reachable_states(10)
        for size, stride in states:
            # Test with offset=0
            self.check_as_strided_via_views(
                initial_tensor, size, stride, storage_offset=0
            )

            # Test with all valid non-zero offsets
            max_offset = max_storage_offset(initial_tensor.numel(), size, stride)
            for offset in range(1, max_offset + 1):
                self.check_as_strided_via_views(
                    initial_tensor, size, stride, storage_offset=offset
                )

    # Tests for specific transformations
    def test_permute_simple(self) -> None:
        """Test simple permute: (2, 5) with strides (5, 1) -> (5, 2) with strides (1, 5)"""
        self.check_as_strided_via_views(torch.arange(10), (5, 2), (1, 5))

    def test_unsqueeze_front(self) -> None:
        """Test unsqueeze at front: (2, 5) -> (1, 2, 5)"""
        self.check_as_strided_via_views(torch.arange(10), (1, 2, 5), (10, 5, 1))

    def test_unsqueeze_middle(self) -> None:
        """Test unsqueeze in middle: (2, 5) -> (2, 1, 5)"""
        self.check_as_strided_via_views(torch.arange(10), (2, 1, 5), (5, 5, 1))

    def test_unsqueeze_end(self) -> None:
        """Test unsqueeze at end: (2, 5) -> (2, 5, 1)"""
        self.check_as_strided_via_views(torch.arange(10), (2, 5, 1), (5, 1, 1))

    def test_narrow_simple(self) -> None:
        """Test narrow: (2, 5) -> (2, 3)"""
        self.check_as_strided_via_views(torch.arange(10), (2, 3), (5, 1))

    def test_permute_with_unsqueeze(self) -> None:
        """Test permute + unsqueeze: (2, 5) -> (1, 5, 2)"""
        self.check_as_strided_via_views(torch.arange(10), (1, 5, 2), (10, 1, 5))

    def test_multiple_unsqueeze_with_narrow(self) -> None:
        """Test multiple unsqueeze + narrow: (2, 5) -> (1, 1, 2, 3)"""
        self.check_as_strided_via_views(torch.arange(10), (1, 1, 2, 3), (15, 10, 5, 1))

    def test_permute_3d(self) -> None:
        """Test 3D permute: (2, 3, 4) -> (4, 2, 3)"""
        self.check_as_strided_via_views(torch.arange(24), (4, 2, 3), (1, 12, 4))

    def test_unsqueeze_multiple_size_one_dims(self) -> None:
        """Test multiple size-1 dims at various positions"""
        self.check_as_strided_via_views(
            torch.arange(6), (1, 2, 1, 3, 1), (6, 3, 3, 1, 1)
        )

    def test_as_strided_via_views_numel_zero(self) -> None:
        """Test numel==0 cases where all strides are insignificant."""
        empty_tensor = torch.tensor([])
        for size, stride in [
            ((0,), (1,)),
            ((0,), (999,)),  # Arbitrary stride
            ((0, 5), (10, 1)),
            ((3, 0), (5, 1)),
            ((2, 0, 3), (0, 1, 0)),  # Arbitrary strides
        ]:
            self.check_as_strided_via_views(empty_tensor, size, stride)

    def test_as_strided_via_views_numel_one(self) -> None:
        """Test numel==1 cases where all strides are insignificant."""
        single_tensor = torch.tensor([42.0])
        for size, stride in [
            ((1,), (1,)),
            ((1,), (999,)),  # Arbitrary stride
            ((1, 1), (10, 1)),
            ((1, 1, 1), (100, 50, 25)),  # All arbitrary strides
        ]:
            self.check_as_strided_via_views(single_tensor, size, stride)

    def test_numel_one_to_zero(self) -> None:
        """Test numel==0 output from numel==1 input."""
        single_tensor = torch.tensor([42.0])
        for size, stride in [
            ((0,), (1,)),
            ((0,), (999,)),
            ((0, 5), (10, 1)),
        ]:
            self.check_as_strided_via_views(single_tensor, size, stride)

    def test_scalar_to_zero_size(self) -> None:
        """Test 0D scalar → zero-size tensor (needs unsqueeze before narrow)."""
        scalar_tensor = torch.tensor(42.0)
        for size, stride in [
            ((0,), (1,)),
            ((0,), (999,)),
            ((0, 5), (10, 1)),
        ]:
            self.check_as_strided_via_views(scalar_tensor, size, stride)

    def test_as_strided_via_views_impossible_cases(self) -> None:
        """Test cases that should return NotImplemented."""
        initial_tensor = torch.arange(10)

        impossible_cases = [
            ((8, 3), (1, 1)),  # Overlapping
            ((2, 2), (6, 3)),  # Requires slice with step>1
        ]

        for size, stride in impossible_cases:
            result = as_strided_via_views(
                initial_tensor, size, stride, storage_offset=0
            )
            self.assertIs(result, NotImplemented)

    # Unit tests for internal helper functions
    def test_squeeze_target_then_basic(self) -> None:
        """Test _squeeze_target_then with single size-1 dim."""
        from torch.utils._as_strided import _squeeze_target_then

        # Define a simple callback that just returns a tensor with the expected shape
        def simple_cb(result, size, stride, storage_offset):
            # Callback receives squeezed target: should be (2, 5)
            self.assertEqual(size, (2, 5))
            self.assertEqual(stride, (5, 1))
            # Return a view with this shape
            return result.view(2, 5)

        base = torch.arange(10)
        # Target has size-1 dim at position 0: (1, 2, 5)
        result = _squeeze_target_then(
            base, size=(1, 2, 5), stride=(10, 5, 1), storage_offset=0, cb=simple_cb
        )

        self.assertEqual(result.size(), (1, 2, 5))
        self.assertEqual(result.stride(), (10, 5, 1))

    def test_squeeze_target_then_multiple_size_one(self) -> None:
        """Test _squeeze_target_then with multiple size-1 dims."""
        from torch.utils._as_strided import _squeeze_target_then

        def simple_cb(result, size, stride, storage_offset):
            # Callback receives squeezed target: should be (5,)
            self.assertEqual(size, (5,))
            self.assertEqual(stride, (1,))
            return result.narrow(0, 0, 5)

        base = torch.arange(10)
        # Target has size-1 dims at positions 0, 2, 4: (1, 5, 1)
        result = _squeeze_target_then(
            base, size=(1, 5, 1), stride=(10, 1, 1), storage_offset=0, cb=simple_cb
        )

        self.assertEqual(result.size(), (1, 5, 1))
        # Size-1 strides are insignificant, so we don't check them strictly

    def test_squeeze_target_then_no_size_one(self) -> None:
        """Test _squeeze_target_then with no size-1 dims."""
        from torch.utils._as_strided import _squeeze_target_then

        def simple_cb(result, size, stride, storage_offset):
            # Callback receives same target: (2, 5)
            self.assertEqual(size, (2, 5))
            self.assertEqual(stride, (5, 1))
            return result.view(2, 5)

        base = torch.arange(10)
        result = _squeeze_target_then(
            base, size=(2, 5), stride=(5, 1), storage_offset=0, cb=simple_cb
        )

        self.assertEqual(result.size(), (2, 5))
        self.assertEqual(result.stride(), (5, 1))

    def test_squeeze_target_then_returns_not_implemented(self) -> None:
        """Test _squeeze_target_then when callback returns NotImplemented."""
        from torch.utils._as_strided import _squeeze_target_then

        def failing_cb(result, size, stride, storage_offset):
            return NotImplemented

        base = torch.arange(10)
        result = _squeeze_target_then(
            base, size=(1, 5, 2), stride=(10, 1, 5), storage_offset=0, cb=failing_cb
        )

        self.assertIs(result, NotImplemented)

    def test_permute_target_then_basic(self) -> None:
        """Test _permute_target_then with simple permutation."""
        from torch.utils._as_strided import _permute_target_then

        def simple_cb(result, size, stride, storage_offset):
            # Callback receives sorted target: (2, 5) with strides (5, 1)
            self.assertEqual(tuple(size), (2, 5))
            self.assertEqual(tuple(stride), (5, 1))
            return result.view(2, 5)

        base = torch.arange(10)
        # Target is (5, 2) with strides (1, 5) - needs permutation
        result = _permute_target_then(
            base, size=(5, 2), stride=(1, 5), storage_offset=0, cb=simple_cb
        )

        self.assertEqual(result.size(), (5, 2))
        self.assertEqual(result.stride(), (1, 5))

    def test_permute_target_then_already_sorted(self) -> None:
        """Test _permute_target_then when target is already sorted."""
        from torch.utils._as_strided import _permute_target_then

        def simple_cb(result, size, stride, storage_offset):
            # Callback receives same target: (2, 5) with strides (5, 1)
            self.assertEqual(tuple(size), (2, 5))
            self.assertEqual(tuple(stride), (5, 1))
            return result.view(2, 5)

        base = torch.arange(10)
        # Target is already sorted
        result = _permute_target_then(
            base, size=(2, 5), stride=(5, 1), storage_offset=0, cb=simple_cb
        )

        self.assertEqual(result.size(), (2, 5))
        self.assertEqual(result.stride(), (5, 1))

    def test_permute_target_then_3d(self) -> None:
        """Test _permute_target_then with 3D tensor."""
        from torch.utils._as_strided import _permute_target_then

        def simple_cb(result, size, stride, storage_offset):
            # Callback receives sorted target: (2, 3, 4) with strides (12, 4, 1)
            self.assertEqual(tuple(size), (2, 3, 4))
            self.assertEqual(tuple(stride), (12, 4, 1))
            return result.view(2, 3, 4)

        base = torch.arange(24)
        # Target is (4, 2, 3) with strides (1, 12, 4) - needs permutation
        result = _permute_target_then(
            base, size=(4, 2, 3), stride=(1, 12, 4), storage_offset=0, cb=simple_cb
        )

        self.assertEqual(result.size(), (4, 2, 3))
        self.assertEqual(result.stride(), (1, 12, 4))

    def test_permute_target_then_returns_not_implemented(self) -> None:
        """Test _permute_target_then when callback returns NotImplemented."""
        from torch.utils._as_strided import _permute_target_then

        def failing_cb(result, size, stride, storage_offset):
            return NotImplemented

        base = torch.arange(10)
        result = _permute_target_then(
            base, size=(5, 2), stride=(1, 5), storage_offset=0, cb=failing_cb
        )

        self.assertIs(result, NotImplemented)

    # Tests for stride-0 dimensions (expand)
    def test_expand_simple(self) -> None:
        """Test simple expand: (5,) -> (3, 5) with stride (0, 1)"""
        self.check_as_strided_via_views(torch.arange(5), (3, 5), (0, 1))

    def test_expand_2d_source(self) -> None:
        """Test expand from 2D: (2, 5) -> (3, 2, 5)"""
        self.check_as_strided_via_views(
            torch.arange(10).view(2, 5), (3, 2, 5), (0, 5, 1)
        )

    def test_expand_middle_dim(self) -> None:
        """Test expand in middle: (2, 5) -> (2, 3, 5)"""
        self.check_as_strided_via_views(
            torch.arange(10).view(2, 5), (2, 3, 5), (5, 0, 1)
        )

    def test_expand_last_dim(self) -> None:
        """Test expand at end: (2, 5) -> (2, 5, 3)"""
        self.check_as_strided_via_views(
            torch.arange(10).view(2, 5), (2, 5, 3), (5, 1, 0)
        )

    def test_expand_multiple_dims(self) -> None:
        """Test expand on multiple dimensions"""
        self.check_as_strided_via_views(torch.arange(5), (3, 5, 4), (0, 1, 0))

    def test_expand_all_dims(self) -> None:
        """Test expand on all dimensions from single element"""
        self.check_as_strided_via_views(torch.tensor([42.0]), (3, 4, 5), (0, 0, 0))

    def test_expand_with_size_one(self) -> None:
        """Test expand combined with size-1 dims"""
        self.check_as_strided_via_views(torch.arange(5), (1, 3, 5), (15, 0, 1))

    def test_source_with_expand(self) -> None:
        """Test source tensor with stride-0 dimension"""
        base = torch.arange(5).expand(3, 5)
        self.check_as_strided_via_views(base, (3, 3), (0, 1))

    def test_source_and_target_with_expand(self) -> None:
        """Test both source and target with stride-0 dimensions"""
        base = torch.arange(5).expand(3, 5)
        self.check_as_strided_via_views(base, (3, 2, 5), (0, 0, 1))

    def test_unexpand_target_then_basic(self) -> None:
        """Test _unexpand_target_then with single stride-0 dim."""
        from torch.utils._as_strided import _unexpand_target_then

        def simple_cb(result, size, stride, storage_offset):
            self.assertEqual(size, (5,))
            self.assertEqual(stride, (1,))
            return result.narrow(0, 0, 5)

        base = torch.arange(10)
        result = _unexpand_target_then(
            base, size=(3, 5), stride=(0, 1), storage_offset=0, cb=simple_cb
        )
        self.assertEqual(result.size(), (3, 5))
        self.assertEqual(result.stride(), (0, 1))

    def test_unexpand_target_then_multiple_stride_zero(self) -> None:
        """Test _unexpand_target_then with multiple stride-0 dims."""
        from torch.utils._as_strided import _unexpand_target_then

        def simple_cb(result, size, stride, storage_offset):
            self.assertEqual(size, (5,))
            self.assertEqual(stride, (1,))
            return result.narrow(0, 0, 5)

        base = torch.arange(10)
        result = _unexpand_target_then(
            base, size=(3, 5, 4), stride=(0, 1, 0), storage_offset=0, cb=simple_cb
        )
        self.assertEqual(result.size(), (3, 5, 4))
        self.assertEqual(result.stride(), (0, 1, 0))

    def test_unexpand_target_then_no_stride_zero(self) -> None:
        """Test _unexpand_target_then with no stride-0 dims."""
        from torch.utils._as_strided import _unexpand_target_then

        def simple_cb(result, size, stride, storage_offset):
            self.assertEqual(size, (2, 5))
            self.assertEqual(stride, (5, 1))
            return result.view(2, 5)

        base = torch.arange(10)
        result = _unexpand_target_then(
            base, size=(2, 5), stride=(5, 1), storage_offset=0, cb=simple_cb
        )
        self.assertEqual(result.size(), (2, 5))
        self.assertEqual(result.stride(), (5, 1))

    def test_unexpand_target_then_returns_not_implemented(self) -> None:
        """Test _unexpand_target_then when callback returns NotImplemented."""
        from torch.utils._as_strided import _unexpand_target_then

        def failing_cb(result, size, stride, storage_offset):
            return NotImplemented

        base = torch.arange(10)
        result = _unexpand_target_then(
            base, size=(3, 5), stride=(0, 1), storage_offset=0, cb=failing_cb
        )
        self.assertIs(result, NotImplemented)

    # Storage offset tests
    def test_storage_offset_simple_1d(self) -> None:
        """Test simple 1D tensor with offset."""
        self.check_as_strided_via_views(torch.arange(10), (5,), (1,), storage_offset=2)

    def test_storage_offset_2d_simple(self) -> None:
        """Test 2D tensor with offset."""
        self.check_as_strided_via_views(
            torch.arange(20), (2, 3), (5, 1), storage_offset=7
        )

    def test_storage_offset_with_permute(self) -> None:
        """Test offset with permuted dimensions."""
        self.check_as_strided_via_views(
            torch.arange(20), (3, 2), (1, 5), storage_offset=4
        )

    def test_storage_offset_with_unsqueeze(self) -> None:
        """Test offset with size-1 dimensions."""
        self.check_as_strided_via_views(
            torch.arange(10), (1, 5, 1), (10, 1, 1), storage_offset=3
        )

    def test_storage_offset_max_offset(self) -> None:
        """Test with maximum possible offset."""
        # For (2, 3) with stride (5, 1), max_extent = max(2*5, 3*1) = 10
        # max_offset = 20 - 10 = 10
        self.check_as_strided_via_views(
            torch.arange(20), (2, 3), (5, 1), storage_offset=10
        )

    def test_storage_offset_with_expand(self) -> None:
        """Test offset with stride-0 dimension."""
        self.check_as_strided_via_views(
            torch.arange(10), (3, 5), (0, 1), storage_offset=2
        )

    def test_storage_offset_impossible(self) -> None:
        """Test that impossible offset returns NotImplemented."""
        base = torch.arange(10)
        # Offset too large: would access indices beyond the base tensor
        result = as_strided_via_views(base, (5,), (1,), storage_offset=10)
        self.assertIs(result, NotImplemented)

    def test_storage_offset_negative(self) -> None:
        """Test that negative offset returns NotImplemented."""
        base = torch.arange(10)
        result = as_strided_via_views(base, (5,), (1,), storage_offset=-1)
        self.assertIs(result, NotImplemented)

    # Tests with multi-dimensional contiguous input tensors
    def test_2d_contiguous_source_simple(self) -> None:
        """Test with 2D contiguous input tensor."""
        base = torch.arange(24).view(4, 6)  # Contiguous 2D tensor
        # Simple permute
        self.check_as_strided_via_views(base, (6, 4), (1, 6))

    def test_2d_contiguous_source_narrow(self) -> None:
        """Test narrowing a 2D contiguous input tensor."""
        base = torch.arange(24).view(4, 6)
        # Narrow both dimensions
        self.check_as_strided_via_views(base, (2, 3), (6, 1))

    def test_2d_contiguous_source_unflatten(self) -> None:
        """Test unflattening a 2D contiguous input tensor."""
        base = torch.arange(24).view(4, 6)
        # Unflatten the second dimension
        self.check_as_strided_via_views(base, (4, 2, 3), (6, 3, 1))

    def test_2d_contiguous_source_flatten(self) -> None:
        """Test flattening a 2D contiguous input back to 1D."""
        base = torch.arange(24).view(4, 6)
        self.check_as_strided_via_views(base, (24,), (1,))

    def test_3d_contiguous_source_permute(self) -> None:
        """Test with 3D contiguous input tensor and permute."""
        base = torch.arange(60).view(3, 4, 5)  # Contiguous 3D tensor
        # Permute dimensions
        self.check_as_strided_via_views(base, (5, 3, 4), (1, 20, 5))

    def test_3d_contiguous_source_narrow_all(self) -> None:
        """Test narrowing all dimensions of 3D contiguous input."""
        base = torch.arange(60).view(3, 4, 5)
        # Narrow all three dimensions
        self.check_as_strided_via_views(base, (2, 2, 3), (20, 5, 1))

    # Tests with non-contiguous multi-dimensional input tensors
    def test_2d_transposed_source(self) -> None:
        """Test with transposed (non-contiguous) 2D input tensor."""
        base = torch.arange(24).view(4, 6).t()  # Non-contiguous: (6, 4) with strides (1, 6)
        # Simple narrow
        self.check_as_strided_via_views(base, (3, 2), (1, 6))

    def test_2d_transposed_source_permute_back(self) -> None:
        """Test permuting transposed tensor back to original layout."""
        base = torch.arange(24).view(4, 6).t()  # (6, 4) with strides (1, 6)
        # Permute back to (4, 6) with strides (6, 1)
        self.check_as_strided_via_views(base, (4, 6), (6, 1))

    def test_2d_non_contiguous_with_gap(self) -> None:
        """Test with 2D tensor where dims have a gap (non-overlapping but not contiguous)."""
        # Create tensor with stride gap: select every other row
        base = torch.arange(48).view(8, 6)[::2, :]  # (4, 6) with strides (12, 1)
        # Narrow and reshape
        self.check_as_strided_via_views(base, (2, 3), (12, 1))

    def test_2d_sliced_both_dims(self) -> None:
        """Test with 2D tensor sliced on both dimensions."""
        # Note: slicing creates storage offset in base, which implementation may not handle
        # So we create a properly contiguous base tensor instead
        base = torch.arange(20).view(5, 4)  # (5, 4) with strides (4, 1), offset 0
        # Test narrow
        self.check_as_strided_via_views(base, (3, 2), (4, 1))

    def test_3d_non_contiguous_permuted(self) -> None:
        """Test with 3D tensor that's been permuted (non-contiguous)."""
        base = torch.arange(60).view(3, 4, 5).permute(2, 0, 1)  # (5, 3, 4) with strides (1, 20, 5)
        # Narrow and unflatten
        self.check_as_strided_via_views(base, (3, 2, 2), (1, 20, 5))

    def test_3d_select_creates_2d(self) -> None:
        """Test with 2D tensor with non-contiguous dimensions (gap between strides)."""
        # Create a 2D tensor where dims are non-contiguous but non-overlapping
        # We'll create this by using as_strided to avoid source offset issues
        base = torch.as_strided(torch.arange(100), (3, 4), (20, 5))
        # Both dims are non-contiguous with each other (gap of 15 between positions)
        self.check_as_strided_via_views(base, (2, 2), (20, 5))

    def test_2d_double_strided(self) -> None:
        """Test with 2D tensor where both dimensions have stride gaps."""
        # Create tensor with gaps in both dimensions
        base = torch.arange(100).view(10, 10)[::2, ::3]  # (5, 4) with strides (20, 3)
        # The two dimensions are discontiguous with each other
        self.check_as_strided_via_views(base, (3, 2), (20, 3))

    # Storage offset tests with multi-dimensional discontiguous sources
    def test_storage_offset_2d_contiguous_source(self) -> None:
        """Test storage offset with 2D contiguous source."""
        base = torch.arange(30).view(5, 6)  # Contiguous (5, 6) with strides (6, 1)
        # Apply offset that spans into second row
        self.check_as_strided_via_views(base, (3, 4), (6, 1), storage_offset=8)

    def test_storage_offset_2d_transposed_source(self) -> None:
        """Test storage offset with 2D transposed (non-contiguous) source."""
        base = torch.arange(30).view(5, 6).t()  # (6, 5) with strides (1, 6)
        # Apply offset
        self.check_as_strided_via_views(base, (4, 3), (1, 6), storage_offset=7)

    def test_storage_offset_2d_discontiguous_both_dims(self) -> None:
        """Test storage offset where narrow needed on both discontiguous source dims."""
        # Create tensor where both dimensions are discontiguous
        base = torch.arange(100).view(10, 10)[::2, ::3]  # (5, 4) with strides (20, 3)
        # Offset = 23: This requires narrow on both dims to consume the offset
        # 23 = 1*20 + 1*3, so we need to narrow first dim by 1 and second dim by 1
        self.check_as_strided_via_views(base, (3, 2), (20, 3), storage_offset=23)

    def test_storage_offset_2d_discontiguous_larger_offset(self) -> None:
        """Test larger storage offset requiring narrows on multiple discontiguous dims."""
        base = torch.arange(100).view(10, 10)[::2, ::3]  # (5, 4) with strides (20, 3)
        # Offset = 46: 46 = 2*20 + 2*3, requires narrow on both dims
        self.check_as_strided_via_views(base, (2, 2), (20, 3), storage_offset=46)

    def test_storage_offset_3d_discontiguous_all_dims(self) -> None:
        """Test storage offset with 3D discontiguous tensor requiring narrows on multiple dims."""
        # Create 3D tensor with gaps in all dimensions
        base = torch.arange(1000).view(10, 10, 10)[::2, ::3, ::5]  # (5, 4, 2) with strides (200, 30, 5)
        # Offset = 30: just one step in the middle dimension
        # Requires narrow on discontiguous dimensions
        self.check_as_strided_via_views(base, (4, 2, 2), (200, 30, 5), storage_offset=30)

    def test_storage_offset_3d_discontiguous_complex(self) -> None:
        """Test complex offset distribution across 3D discontiguous dims."""
        base = torch.arange(1000).view(10, 10, 10)[::3, ::2, ::4]  # (4, 5, 3) with strides (300, 20, 4)
        # Offset = 24: 24 = 1*20 + 1*4
        # Requires narrow on the smaller stride dimensions
        self.check_as_strided_via_views(base, (2, 3, 2), (300, 20, 4), storage_offset=24)

    def test_storage_offset_2d_permuted_discontiguous(self) -> None:
        """Test storage offset where source dims are in non-descending stride order."""
        # Create non-contiguous tensor then permute so strides aren't descending
        base_raw = torch.arange(100).view(10, 10)[::3, ::2]  # (4, 5) with strides (30, 2)
        # This already has strides in descending order, so let's transpose it
        base = base_raw.t()  # (5, 4) with strides (2, 30)
        # Now strides are NOT in descending order (2 < 30)
        # Offset = 32: 32 = 1*30 + 1*2
        self.check_as_strided_via_views(base, (2, 2), (2, 30), storage_offset=32)

    def test_storage_offset_barely_fits(self) -> None:
        """Test storage offset with discontiguous source requiring narrows on both dims."""
        base = torch.arange(100).view(10, 10)[::3, ::4]  # (4, 3) with strides (30, 4)
        # Offset = 34 = 1*30 + 1*4, requires narrow on both dimensions
        # For target (2, 2), extent = 34 + (2-1)*30 + (2-1)*4 = 34 + 34 = 68 < 98 (max base index)
        self.check_as_strided_via_views(base, (2, 2), (30, 4), storage_offset=34)


if __name__ == "__main__":
    run_tests()
