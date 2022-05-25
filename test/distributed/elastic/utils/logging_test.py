#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch.distributed.elastic.utils.logging as logging
from torch.testing._internal.common_utils import run_tests, TestCase

log = logging.get_logger()


class LoggingTest(TestCase):
    def setUp(self):
        super().setUp()
        self.clazz_log = logging.get_logger()

    def test_logger_name(self):
        local_log = logging.get_logger()
        name_override_log = logging.get_logger("foobar")

        self.assertEqual(__name__, log.name)
        self.assertEqual(__name__, self.clazz_log.name)
        self.assertEqual(__name__, local_log.name)
        self.assertEqual("foobar", name_override_log.name)

    def test_derive_module_name(self):
        module_name = logging._derive_module_name(depth=1)
        self.assertEqual(__name__, module_name)


if __name__ == "__main__":
    run_tests()
