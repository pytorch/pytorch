# Owner(s): ["oncall: distributed"]
import copy
import itertools
import sys
import unittest

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
            (strided_shard, True, False, False),
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
            (Shard(3), _StridedShard(dim=3, split_factor=7)),
            (Shard(4), _StridedShard(dim=4, split_factor=9)),
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

        # Testing this case doesn't seem to fit neatly into the above equivalence class
        # framework.
        self.assertNotEqual(
            _StridedShard(dim=3, split_factor=1), _StridedShard(dim=3, split_factor=2)
        )

    @unittest.skipIf(
        sys.version_info < (3, 10), "kw_only is only available in python >= 3.10"
    )
    def test_strided_shard_kwonly_argument(self):
        with self.assertRaises(TypeError):
            _StridedShard(3, 4)
        _StridedShard(3, split_factor=4)

    def test_strided_shard_isinstance_shard(self):
        assert isinstance(_StridedShard(dim=3, split_factor=7), Shard)

    def test_dynamo_can_identify_placement_classes(self):
        for cls in (Replicate, Shard, _StridedShard, Partial):
            self.assertTrue(
                PlacementClassVariable.is_placement_type(cls), msg=f"failed on {cls}"
            )


if __name__ == "__main__":
    run_tests()
