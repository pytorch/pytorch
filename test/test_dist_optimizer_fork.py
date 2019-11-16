#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from dist_optimizer_test import DistOptimizerTest
from common_distributed import MultiProcessTestCase
from common_utils import run_tests


class DistOptimizerTestWithFork(MultiProcessTestCase, DistOptimizerTest):

    def setUp(self):
        super(DistOptimizerTestWithFork, self).setUp()
        self._fork_processes()

if __name__ == '__main__':
    run_tests()
