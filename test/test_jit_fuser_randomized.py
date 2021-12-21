# Owner(s): ["NNC"]

import torch
import torch.nn.functional as F
import os

# these needs to be set before `common_utils`
# infers `GRAPH_EXECUTOR`.
# this file **requires** these settings
# and setting them after `GRAPH_EXECUTOR` is
# inferred erroneously runs or skips
# some tests
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)


from torch.testing._internal.common_utils import run_tests, TestCase


class TestRandomized(TestCase):
    def setUp(self):
        # Set the seed to 1. This tests the codepath through random
        # transformation.
        os.environ["PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED"] = "1"

    def tearDown(self):
        # Set it back to 0.
        os.environ["PYTORCH_TENSOREXPR_RANDOM_TRANSFORM_SEED"] = "0"

    def test_relu(self):
        def fn_test_relu(x, y):
            return F.relu(x + 0.5 * y)

        device = "cpu"
        x = torch.randn(4, 4, dtype=torch.float, device=device)
        y = torch.randn(4, 4, dtype=torch.float, device=device)

        fn = fn_test_relu
        traced_fn = torch.jit.trace(fn, (x, y))

        ref = fn(x, y)
        res = traced_fn(x, y)
        assert torch.allclose(ref, res)


if __name__ == "__main__":
    run_tests()
