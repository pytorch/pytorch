#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# Owner(s): ["oncall: aiacc"]

# Test that this import should not trigger any error when run
# in non-GPU hosts, or in any build mode.
import fx2trt_oss.fx.lower as fxl  # noqa: F401
from torch.testing._internal.common_utils import TestCase, run_tests


class MainTests(TestCase):
    def test_1(self):
        pass

if __name__ == '__main__':
    run_tests()
