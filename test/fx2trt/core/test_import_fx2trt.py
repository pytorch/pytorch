#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import unittest as ut

# Test that this import should not trigger any error when run
# in non-GPU hosts, or in any build mode.
import torch.fx.experimental.fx2trt.lower as fxl  # noqa: F401


class MainTests(ut.TestCase):
    def test_1(self):
        pass
