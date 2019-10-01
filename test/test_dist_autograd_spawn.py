#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from dist_autograd_test import DistAutogradTest
from common_distributed import MultiProcessTestCase
from common_utils import run_tests

import unittest

@unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/27157")
class DistAutogradTestWithSpawn(MultiProcessTestCase, DistAutogradTest):

    def setUp(self):
        super(DistAutogradTestWithSpawn, self).setUp()
        self._spawn_processes()

if __name__ == '__main__':
    run_tests()
