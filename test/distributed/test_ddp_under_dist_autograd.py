#!/usr/bin/env python3

from torch.testing._internal.distributed import ddp_under_dist_autograd_test
from torch.testing._internal.common_utils import (
    run_tests,
)

class TestDdpUnderDistAutogradWrapper(ddp_under_dist_autograd_test.TestDdpUnderDistAutograd):
    pass

class TestDdpComparison(ddp_under_dist_autograd_test.TestDdpComparison):
    pass

if __name__ == "__main__":
    run_tests()
