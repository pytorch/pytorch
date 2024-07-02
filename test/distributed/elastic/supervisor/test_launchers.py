#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

import torch.distributed.elastic.supervisor.launchers as launchers

import torch.testing._internal.common_utils as testing_common

try:
    import zmq  # noqa: F401
except ImportError:
    raise unittest.SkipTest("zmq not installed in test harness") from None


def path(script):
    return os.path.join(os.path.dirname(__file__), script)


class LaunchersTest(testing_common.TestCase):
    def test_supervisor_registry(self):
        self.assertEqual(
            launchers.launcher_registry["default"], launchers._default_launcher
        )


if __name__ == "__main__":
    testing_common.run_tests()
