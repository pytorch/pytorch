# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.utils.meta_init import (
    clear_meta_init_cache,
    is_meta_init,
    materialize,
    meta_init,
)
from torch.testing._internal.common_utils import TestCase, run_tests


class MetaInitTest(TestCase):
    def test_meta_init(self) -> None:
        x = torch.ones([2, 2])

        with meta_init():
            a = torch.ones([2, 2])
            b = torch.ones([2, 2], device="cpu")
            c = torch.ones([2, 2], device="meta")
            d = a + b + c
            d.add_(3)
            e = a.view(-1)
            f = x + a

            g = torch.arange(10)

            y = torch.tensor(1.)
            z = y + 2

        meta_device = torch.device("meta")

        self.assertEqual(a.device, meta_device)
        self.assertEqual(b.device, meta_device)
        self.assertEqual(c.device, meta_device)
        self.assertEqual(d.device, meta_device)
        self.assertEqual(e.device, meta_device)
        self.assertEqual(f.device, meta_device)
        self.assertEqual(g.device, meta_device)
        self.assertEqual(y.device, torch.device("cpu"))
        self.assertEqual(z.device, meta_device)

        self.assertEqual(a.shape, [2, 2])
        self.assertEqual(b.shape, [2, 2])
        self.assertEqual(c.shape, [2, 2])
        self.assertEqual(d.shape, [2, 2])
        self.assertEqual(e.shape, [4])
        self.assertEqual(f.shape, [2, 2])
        self.assertEqual(g.shape, [10])
        self.assertEqual(y.shape, [])
        self.assertEqual(z.shape, [])

    def test_is_meta_init_returns_correct_value(self) -> None:
        self.assertFalse(is_meta_init())

        with meta_init():
            self.assertTrue(is_meta_init())

            with meta_init():
                self.assertTrue(is_meta_init())

            self.assertTrue(is_meta_init())

        self.assertFalse(is_meta_init())

    def test_materialize_is_noop(self) -> None:
        with meta_init():
            tensor = torch.ones([10, 10])

        tensor_id = id(tensor)

        materialize(tensor)

        self.assertEqual(tensor_id, id(tensor))

    def test_clear_meta_init_cache_is_noop(self):
        clear_meta_init_cache()


if __name__ == "__main__":
    run_tests()
