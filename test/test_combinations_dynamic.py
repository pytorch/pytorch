# Verifies fix for symbolic numel in aten::combinations shape logic.
# See: https://github.com/pytorch/pytorch/issues/163759
import unittest
import torch

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests
)

class TestCombinationsDynamic(TestCase):
    def _eager(self, x, r, with_replacement):
        out = torch.combinations(x.flatten(), r=r, with_replacement=with_replacement)
        # Canonicalize for stable comparison
        return out.to(torch.float32).sort(dim=0).values

    def _compiled(self, r, with_replacement):
        def fn(x):
            return torch.combinations(x.flatten(), r=r, with_replacement=with_replacement)
        # The original bug repro failed under aot_eager + dynamic=True
        return torch.compile(fn, backend="aot_eager", dynamic=True)

    def _assert_match(self, compiled, x, r, with_replacement):
        out = compiled(x)
        exp = self._eager(x, r=r, with_replacement=with_replacement)
        self.assertEqual(out.to(torch.float32).sort(dim=0).values, exp)

    def test_dynamic_shapes_r2_matches_eager(self):
        # Runs twice with different input sizes to exercise dynamic shapes
        for with_replacement in (False, True):
            compiled = self._compiled(r=2, with_replacement=with_replacement)
            self._assert_match(compiled, torch.tensor([1, 2, 3, 4], dtype=torch.int64), r=2, with_replacement=with_replacement)
            self._assert_match(compiled, torch.tensor([5, 6, 7], dtype=torch.int64), r=2, with_replacement=with_replacement)

    def test_dynamic_shapes_r3_small_input(self):
        # A small additional case to guard future regressions
        compiled = self._compiled(r=3, with_replacement=False)
        self._assert_match(compiled, torch.tensor([1, 2, 3, 4], dtype=torch.int64), r=3, with_replacement=False)

if __name__ == "__main__":
    run_tests()

