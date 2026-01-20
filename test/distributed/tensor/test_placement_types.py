# Owner(s): ["oncall: distributed"]
import copy
import itertools

import torch
from torch._dynamo.variables.distributed import PlacementClassVariable
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# Basic functionality test for Placement types.
class PlacementTypesTestCase(TestCase):
    def test_type_identification(self):
        shard = Shard(3)
        strided_shard = _StridedShard(dim=3, split_factor=7)
        partial_sum = Partial("sum")
        partial_max = Partial("max")
        replicate = Replicate()

        ident_tests = (
            (shard, True, False, False),
            (strided_shard, False, False, False),
            (partial_sum, False, True, False),
            (partial_max, False, True, False),
            (replicate, False, False, True),
        )
        for do_deepcopy in (False, True):
            for placement, is_shard, is_partial, is_replicate in ident_tests:
                if do_deepcopy:
                    placement = copy.deepcopy(placement)
                self.assertEqual(placement.is_shard(), is_shard)
                self.assertEqual(placement.is_partial(), is_partial)
                self.assertEqual(placement.is_replicate(), is_replicate)

    def test_equality(self):
        equivalence_classes = (
            (Shard(3),),
            (Shard(4),),
            (_StridedShard(dim=3, split_factor=1),),
            (_StridedShard(dim=3, split_factor=2),),
            (_StridedShard(dim=4, split_factor=9),),
            (Replicate(),),
            (Partial("sum"),),
            (Partial("max"),),
        )
        for eq_class in equivalence_classes:
            # Each item in the equivalence class should be equal to every other item in
            # its class.
            for lhs, rhs in itertools.product(eq_class, eq_class):
                self.assertEqual(lhs, rhs)

            # Each item in the equivalence class should not be equal to any item in any
            # other class.
            for other_class in equivalence_classes:
                if other_class is eq_class:
                    continue
                for lhs, rhs in itertools.product(eq_class, other_class):
                    self.assertNotEqual(lhs, rhs)

    def test_strided_shard_kwonly_argument(self):
        with self.assertRaises(TypeError):
            _StridedShard(3, 4)
        _StridedShard(3, split_factor=4)

    def test_dynamo_can_identify_placement_classes(self):
        for cls in (Replicate, Shard, _StridedShard, Partial):
            self.assertTrue(
                PlacementClassVariable.is_placement_type(cls), msg=f"failed on {cls}"
            )

    def test_select_split_tensor_matches_split_tensor(self):
        """
        Test that _select_split_tensor produces the same result as indexing
        into _split_tensor. This validates that any alternative implementation
        (e.g., the narrow-based SymInt path) matches the canonical _split_tensor.
        """
        # Test various tensor sizes and num_chunks combinations
        test_cases = [
            # (dim_size, num_chunks) - covers even splits, uneven splits, edge cases
            (16, 4),  # even split
            (17, 4),  # uneven split, last chunk smaller
            (15, 4),  # uneven split
            (7, 4),  # fewer elements than chunks would like
            (3, 4),  # very few elements
            (1, 4),  # single element
            (8, 1),  # single chunk
            (8, 8),  # one element per chunk
            (8, 16),  # more chunks than elements
        ]

        for dim in [0, 1]:
            shard = Shard(dim)
            for dim_size, num_chunks in test_cases:
                # Create a tensor with distinct values for easy debugging
                if dim == 0:
                    tensor = torch.arange(dim_size * 4).reshape(dim_size, 4)
                else:
                    tensor = torch.arange(4 * dim_size).reshape(4, dim_size)

                # Get ground truth from _split_tensor
                shards, _ = shard._split_tensor(
                    tensor, num_chunks, with_padding=False, contiguous=False
                )

                # Compare _select_split_tensor against _split_tensor for each index
                for idx in range(num_chunks):
                    selected = shard._select_split_tensor(
                        tensor,
                        num_chunks,
                        idx,
                        with_padding=False,
                        contiguous=False,
                        clone=False,
                    )
                    self.assertEqual(
                        selected,
                        shards[idx],
                        msg=f"Mismatch for dim={dim}, dim_size={dim_size}, "
                        f"num_chunks={num_chunks}, idx={idx}",
                    )

    def test_select_split_tensor_symint_with_padding_raises(self):
        """
        Test that _select_split_tensor raises GuardOnDataDependentSymNode when
        called with a SymInt index and with_padding=True.

        This is expected because with_padding=True requires indexing into a
        Python list with the SymInt, which is not supported.
        """
        from torch.fx.experimental.symbolic_shapes import (
            GuardOnDataDependentSymNode,
            ShapeEnv,
        )

        shape_env = ShapeEnv()
        symint_index = shape_env.create_unbacked_symint()

        shard = Shard(0)
        tensor = torch.arange(16).reshape(4, 4)

        with self.assertRaises(GuardOnDataDependentSymNode):
            shard._select_split_tensor(
                tensor,
                num_chunks=4,
                index=symint_index,
                with_padding=True,
            )


if __name__ == "__main__":
    run_tests()
