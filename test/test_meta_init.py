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
