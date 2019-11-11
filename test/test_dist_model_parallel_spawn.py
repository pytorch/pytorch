#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from dist_model_parallel_test import DistModelParallelTest
from common_distributed import MultiProcessTestCase
from common_utils import TEST_WITH_ASAN, run_tests

import unittest

@unittest.skipIf(TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues")
class DistModelParallelSpawn(MultiProcessTestCase, DistModelParallelTest):

    def setUp(self):
        super(DistModelParallelSpawn, self).setUp()
        self._spawn_processes()

if __name__ == '__main__':
    run_tests()
