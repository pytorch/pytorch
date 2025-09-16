# Owner(s): ["oncall: distributed"]
import itertools
import sys
import unittest

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
        for placement, is_shard, is_partial, is_replicate in ident_tests:
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


if __name__ == "__main__":
    run_tests()
