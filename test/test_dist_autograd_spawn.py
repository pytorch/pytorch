#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from dist_autograd_test import DistAutogradTest
from common_distributed import MultiProcessTestCase
from common_utils import TEST_WITH_ASAN, run_tests

import unittest

class DistAutogradTestWithSpawn(MultiProcessTestCase, DistAutogradTest):

    def setUp(self):
        super(DistAutogradTestWithSpawn, self).setUp()
        self._spawn_processes()

if __name__ == '__main__':
    run_tests()
