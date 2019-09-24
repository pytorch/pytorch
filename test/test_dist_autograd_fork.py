#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from dist_autograd_test import TestDistAutograd
from common_distributed import MultiProcessTestCase
from common_utils import run_tests


class TestDistAutogradWithFork(MultiProcessTestCase, TestDistAutograd):

    def setUp(self):
        super(TestDistAutogradWithFork, self).setUp()
        self._fork_processes()

if __name__ == '__main__':
    run_tests()
