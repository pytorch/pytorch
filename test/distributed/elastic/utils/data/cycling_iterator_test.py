#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from torch.distributed.elastic.utils.data import CyclingIterator


class CyclingIteratorTest(unittest.TestCase):
    def generator(self, epoch, stride, max_epochs):
        # generate an continuously incrementing list each epoch
        # e.g. [0,1,2] [3,4,5] [6,7,8] ...
        return iter([stride * epoch + i for i in range(0, stride)])

    def test_cycling_iterator(self):
        stride = 3
        max_epochs = 90

        def generator_fn(epoch):
            return self.generator(epoch, stride, max_epochs)

        it = CyclingIterator(n=max_epochs, generator_fn=generator_fn)
        for i in range(0, stride * max_epochs):
            self.assertEqual(i, next(it))

        with self.assertRaises(StopIteration):
            next(it)

    def test_cycling_iterator_start_epoch(self):
        stride = 3
        max_epochs = 2
        start_epoch = 1

        def generator_fn(epoch):
            return self.generator(epoch, stride, max_epochs)

        it = CyclingIterator(max_epochs, generator_fn, start_epoch)
        for i in range(stride * start_epoch, stride * max_epochs):
            self.assertEqual(i, next(it))

        with self.assertRaises(StopIteration):
            next(it)
