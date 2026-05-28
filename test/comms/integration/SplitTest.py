#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from integration.helpers.TorchCommTestHelpers import TorchCommTestWrapper

import torch
from torch.comms import ReduceOp


class SplitTest(unittest.TestCase):
    """Test class for split operations in TorchComm."""

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    # Helper function to verify communication within a communicator
    def _verify_communication(self, comm):
        """Verify communication within a communicator."""
        # Skip if communicator is null
        if comm is None:
            return

        # Get rank and size from the communicator
        rank = comm.get_rank()
        size = comm.get_size()

        # Create input tensor
        options = {"dtype": torch.float, "device": self.device}
        input_tensor = torch.ones(10, **options) * float(rank + 1)

        # For ranks in groups, test all_reduce
        comm.all_reduce(input_tensor, ReduceOp.SUM, False)

        # Verify the result using tensor equality with integer input
        expected_sum = size * (size + 1) / 2
        self._verify_tensor_equality(input_tensor.cpu(), expected_sum)

    # Helper function to create contiguous groups with even distribution
    def _create_contig_groups(self, num_ranks, num_groups, empty_groups):
        """Create contiguous groups with even distribution."""
        rank_groups = [[] for _ in range(num_groups)]

        # Calculate base size for each group
        base_size = num_ranks // num_groups
        # Calculate how many groups get an extra rank
        remainder = num_ranks % num_groups

        rank_index = 0

        # Create each group
        for group in range(num_groups):
            # Calculate size for this group
            group_size = base_size
            # Add extra rank to last group
            if group == num_groups - 1:
                group_size += remainder

            # If this group should be empty, skip the ranks that would have been in it
            if group in empty_groups:
                rank_index += group_size  # Skip these ranks
                continue

            # Add ranks to this group
            for _ in range(group_size):
                if rank_index < num_ranks:
                    rank_groups[group].append(rank_index)
                    rank_index += 1

        return rank_groups

    # Helper function to create non-contiguous groups with round-robin distribution
    def _create_noncontig_groups(self, num_ranks, num_groups, empty_groups):
        """Create non-contiguous groups with round-robin distribution."""
        rank_groups = [[] for _ in range(num_groups)]

        # Distribute ranks in round-robin manner
        for rank in range(num_ranks):
            group = rank % num_groups

            # Skip empty groups
            if group not in empty_groups:
                rank_groups[group].append(rank)

        return rank_groups

    # Helper function to verify contiguous groups
    def _verify_contig_groups(self, parent_comm, child_comm, num_groups, empty_groups):
        """Verify contiguous groups."""
        parent_rank = parent_comm.get_rank()
        parent_size = parent_comm.get_size()

        # Use _create_contig_groups to get the rank groups
        rank_groups = self._create_contig_groups(parent_size, num_groups, empty_groups)

        # Determine which group this rank belongs to
        my_group = -1
        rank_in_group = -1
        for group in range(num_groups):
            # Check if parent_rank is in this group
            if parent_rank in rank_groups[group]:
                my_group = group
                rank_in_group = rank_groups[group].index(parent_rank)
                break

        # Verify child_comm based on whether this rank is in an empty group
        if my_group == -1:
            # This rank should not be in any group
            self.assertIsNone(
                child_comm, f"Rank {parent_rank} should not have a child communicator"
            )
        else:
            # This rank should be in a non-empty group
            self.assertIsNotNone(
                child_comm, f"Rank {parent_rank} should have a child communicator"
            )

            # Verify child rank
            self.assertEqual(
                child_comm.get_rank(),
                rank_in_group,
                f"Incorrect child rank for parent rank {parent_rank}",
            )

    # Helper function to verify non-contiguous groups
    def _verify_noncontig_groups(
        self, parent_comm, child_comm, num_groups, empty_groups
    ):
        """Verify non-contiguous groups."""
        parent_rank = parent_comm.get_rank()
        parent_size = parent_comm.get_size()

        # Use _create_noncontig_groups to get the rank groups
        rank_groups = self._create_noncontig_groups(
            parent_size, num_groups, empty_groups
        )

        # Determine which group this rank belongs to
        my_group = -1
        rank_in_group = -1
        for group in range(num_groups):
            # Check if parent_rank is in this group
            if parent_rank in rank_groups[group]:
                my_group = group
                rank_in_group = rank_groups[group].index(parent_rank)
                break

        # Verify child_comm based on whether this rank is in an empty group
        if my_group == -1:
            # This rank should not be in any group
            self.assertIsNone(
                child_comm, f"Rank {parent_rank} should not have a child communicator"
            )
        else:
            # This rank should be in a non-empty group
            self.assertIsNotNone(
                child_comm, f"Rank {parent_rank} should have a child communicator"
            )

            # Verify child rank
            self.assertEqual(
                child_comm.get_rank(),
                rank_in_group,
                f"Incorrect child rank for parent rank {parent_rank}",
            )

    def _verify_tensor_equality(self, output, expected_value, description=""):
        """Verify tensor equality with appropriate comparison."""
        # Skip verification if tensor is empty
        if output.numel() == 0:
            return

        # Create expected tensor with the same size and dtype as output
        if isinstance(expected_value, (int, float)):
            expected = torch.full_like(output, float(expected_value))
        else:
            expected = expected_value

        # Check that tensors have the same shape
        self.assertTrue(
            output.size() == expected.size(),
            f"Tensor shapes don't match for {description}",
        )

        # Check that tensors have the same dtype
        self.assertEqual(
            output.dtype, expected.dtype, f"Tensor dtypes don't match for {description}"
        )

        # Different verification based on dtype
        if output.dtype == torch.float:
            # For float tensors, check if they are close enough
            diff = torch.abs(output - expected)
            all_close = diff.max().item() < 1e-5
            self.assertTrue(
                all_close, f"Tensors are not close enough for {description}"
            )
        else:
            # For integer types, check exact equality
            equal = torch.all(output.eq(expected)).item()
            self.assertTrue(equal, f"Tensors are not equal for {description}")

    def test_contiguous_group(self):
        """Test contiguous group with first half of ranks."""
        split_size = self.num_ranks // 2
        if split_size == 0:
            split_size = 1  # Ensure at least one rank

        rank_in_group = self.rank < split_size
        ranks = []

        if rank_in_group:
            # Only fill ranks if current rank is in the group
            ranks.extend(range(split_size))
        # Otherwise, ranks remains empty

        # Call split function
        new_torchcomm = self.torchcomm.split(ranks, name="contiguous_split_comm")

        if rank_in_group:
            # Current rank should be in the group and get a communicator
            self.assertIsNotNone(
                new_torchcomm,
                f"Expected communicator but got None for rank {self.rank}",
            )

            # Verify rank and size
            self.assertEqual(
                new_torchcomm.get_rank(),
                self.rank,
                "New rank should match position in ranks list",
            )
            self.assertEqual(
                new_torchcomm.get_size(),
                split_size,
                "Size should match ranks list size",
            )

            # Test communication within the child communicator
            self._verify_communication(new_torchcomm)

            # Finalize the communicator before it's destroyed
            new_torchcomm.finalize()
        else:
            # Current rank should not be in the group and get None
            self.assertIsNone(
                new_torchcomm, f"Rank {self.rank} should not have a child communicator"
            )

    def test_non_contiguous_group(self):
        """Test non-contiguous group with even ranks only."""
        rank_in_group = self.rank % 2 == 0  # Even ranks only
        ranks = []

        if rank_in_group:
            # Only fill ranks if current rank is in the group (even ranks)
            ranks.extend(range(0, self.num_ranks, 2))
        # Otherwise, ranks remains empty

        # Call split function
        new_torchcomm = self.torchcomm.split(ranks, name="noncontig_child_comm")

        if rank_in_group:
            # Current rank should be in the group and get a communicator
            self.assertIsNotNone(
                new_torchcomm,
                f"Expected communicator but got None for rank {self.rank}",
            )

            # Verify rank and size
            expected_new_rank = self.rank // 2  # Position among even ranks
            expected_size = (self.num_ranks + 1) // 2  # Number of even ranks
            self.assertEqual(
                new_torchcomm.get_rank(),
                expected_new_rank,
                "New rank should match position in even ranks",
            )
            self.assertEqual(
                new_torchcomm.get_size(),
                expected_size,
                "Size should match number of even ranks",
            )

            # Test communication within the child communicator
            self._verify_communication(new_torchcomm)

            # Finalize the communicator before it's destroyed
            new_torchcomm.finalize()
        else:
            # Current rank should not be in the group and get None
            self.assertIsNone(
                new_torchcomm, f"Rank {self.rank} should not have a child communicator"
            )

    def test_duplicate_ranks(self):
        """Test that duplicate ranks are properly rejected with a runtime error."""
        # All ranks pass the same duplicate ranks list to ensure collective validation
        # Include all ranks but duplicate them to trigger validation error
        ranks_with_duplicates = list(range(self.num_ranks)) + list(
            range(self.num_ranks)
        )

        # Call split function - this should throw a RuntimeError due to duplicate ranks
        # All ranks should get the same exception since duplicate detection should happen
        # during the collective validation phase
        with self.assertRaises(RuntimeError):
            self.torchcomm.split(ranks_with_duplicates, name="duplicate_comm")

    def test_rank_not_in_group(self):
        """Test that a rank not including itself in the group is rejected with a runtime error."""
        # Each rank passes a list containing all other ranks except itself
        # This should trigger a runtime error since a rank cannot participate in a split
        # operation without including itself in the ranks list
        ranks_excluding_self = [r for r in range(self.num_ranks) if r != self.rank]

        # Call split function - this should throw a RuntimeError since current rank
        # is not included in the ranks list it's trying to split with
        with self.assertRaises(RuntimeError):
            self.torchcomm.split(ranks_excluding_self, name="exclude_self_comm")

    def test_multi_level(self):
        """Test multi-level split."""
        # First level split: Include first half of ranks
        first_split_size = self.num_ranks // 2
        if first_split_size == 0:
            first_split_size = 1

        rank_in_first_level = self.rank < first_split_size
        first_level_ranks = []

        if rank_in_first_level:
            # Only fill ranks if current rank is in the first level
            first_level_ranks.extend(range(first_split_size))
        # Otherwise, first_level_ranks remains empty

        # Call split function to create first-level child communicator
        first_level_comm = self.torchcomm.split(
            first_level_ranks, name="first_level_comm"
        )

        # Test communication on parent communicator first (all ranks participate)
        options = {"dtype": torch.float, "device": self.device}
        parent_input = torch.ones(10, **options) * float(self.rank + 1)
        self.torchcomm.all_reduce(parent_input, ReduceOp.SUM, False)
        self._verify_tensor_equality(
            parent_input.cpu(), self.num_ranks * (self.num_ranks + 1) / 2
        )

        if not rank_in_first_level:
            # Current rank is not in first level, should get None
            self.assertIsNone(
                first_level_comm,
                f"Rank {self.rank} should not have a first-level child communicator",
            )
            return

        # Current rank is in the first level, should have a communicator
        self.assertIsNotNone(
            first_level_comm,
            f"Expected first-level communicator but got None for rank {self.rank}",
        )

        # Get rank and size from the first-level communicator
        first_level_rank = first_level_comm.get_rank()
        first_level_size = first_level_comm.get_size()

        # Second level split: Split first half of first-level communicator
        second_split_size = first_level_size // 2
        if second_split_size == 0:
            second_split_size = 1

        rank_in_second_level = first_level_rank < second_split_size
        second_level_ranks = []

        if rank_in_second_level:
            # Only fill ranks if current rank is in the second level
            second_level_ranks.extend(range(second_split_size))
        # Otherwise, second_level_ranks remains empty

        # Call split function to create second-level child communicator
        second_level_comm = first_level_comm.split(
            second_level_ranks, name="second_level_comm"
        )

        # Test communication on first-level communicator (only first-level ranks participate)
        first_level_input = torch.ones(10, **options) * float(first_level_rank + 1)
        first_level_comm.all_reduce(first_level_input, ReduceOp.SUM, False)
        self._verify_tensor_equality(
            first_level_input.cpu(), first_level_size * (first_level_size + 1) / 2
        )

        if not rank_in_second_level:
            # Current rank is not in second level, test only with first level
            self.assertIsNone(
                second_level_comm,
                f"Rank {first_level_rank} should not have a second-level child communicator",
            )

            first_level_comm.finalize()
            return

        # Current rank is in the second level, should have a communicator
        self.assertIsNotNone(
            second_level_comm,
            f"Expected second-level communicator but got None for first-level rank {first_level_rank}",
        )

        # Test communication on second-level communicator (only second-level ranks participate)
        second_level_rank = second_level_comm.get_rank()
        second_level_size = second_level_comm.get_size()
        second_level_input = torch.ones(10, **options) * float(second_level_rank + 1)
        second_level_comm.all_reduce(second_level_input, ReduceOp.SUM, False)
        self._verify_tensor_equality(
            second_level_input.cpu(), second_level_size * (second_level_size + 1) / 2
        )

        # Finalize the communicators before they're destroyed
        second_level_comm.finalize()
        first_level_comm.finalize()


if __name__ == "__main__":
    unittest.main(failfast=True)
