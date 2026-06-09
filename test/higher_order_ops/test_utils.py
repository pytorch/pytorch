# Owner(s): ["module: higher order operators"]

import torch
from torch._higher_order_ops.utils import _has_potential_branch_input_mutation
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import run_tests, TestCase


class TestHigherOrderOpUtils(TestCase):
    def test_alias_mutation_query_ignores_discarded_unbacked_symbols(self):
        def branch(x):
            torch.nonzero(x)
            return x + 1

        self.assertFalse(_has_potential_branch_input_mutation(branch, [torch.ones(4)]))

    def test_alias_mutation_query_uses_existing_fake_mode(self):
        def branch(x):
            torch.nonzero(x)
            return x + 1

        with FakeTensorMode(shape_env=ShapeEnv()):
            x = torch.ones(4)
            self.assertFalse(_has_potential_branch_input_mutation(branch, [x]))


if __name__ == "__main__":
    run_tests()
