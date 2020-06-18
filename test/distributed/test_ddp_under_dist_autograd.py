#!/usr/bin/env python3

from torch.testing._internal.distributed.ddp_under_dist_autograd_test import TestDdpUnderDistAutograd
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import TestDdpComparison

if __name__ == "__main__":
    run_tests()
