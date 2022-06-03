# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)

class CheckpointWrapperTest(TestCase):
    def setUp(self):
        super().setUp()

    def test_load_activation_checkpointed_module(self):
        lin = nn.Linear(10, 10, bias=False)
        lin = checkpoint_wrapper(lin)
        state_dict = deepcopy(lin.state_dict())
        # Load into non-checkpoint wrapped linear module
        lin_new = nn.Linear(10, 10, bias=False)
        lin_new.load_state_dict(state_dict)
        for p1, p2 in zip(lin.parameters(), lin_new.parameters()):
            self.assertEqual(p1, p2)
            self.assertTrue(torch.allclose(p1, p2))

        # Load non-checkpoint wrapped module into checkpoint wrapped one
        # Make params different
        for p in lin_new.parameters():
            with torch.no_grad():
                p.add_(0.5)

        state_dict = deepcopy(lin_new.state_dict())
        # Verify checkpoint wrapped linear can load unwrapped linear
        lin.load_state_dict(state_dict)
        for p1, p2 in zip(lin.parameters(), lin_new.parameters()):
            self.assertEqual(p1, p2)


if __name__ == "__main__":
    run_tests()
