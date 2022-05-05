# Owner(s): ["module: primTorch"]

from functools import partial

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    dtypes,
)
import torch._prims as prims
from torch._prims.executor import make_traced


class TestPrims(TestCase):
    @onlyCUDA
    @dtypes(torch.float32)
    def test_broadcast_in_dim(self, device, dtype):
        def _wrapper(a, shape, broadcast_dimensions):
            return prims.broadcast_in_dim(a, shape, broadcast_dimensions)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        # TODO: FIXME:
        # for executor in ('aten', 'nvfuser'):
        for executor in ("aten",):
            fn = partial(traced, executor=executor)
            # Same shape
            shape = (5, 5)
            a = make_arg(shape)
            result = fn(a, shape, (0, 1))

            self.assertEqual(result.shape, a.shape)
            self.assertTrue(result.is_contiguous)
            self.assertEqual(a, result)

            # Error input: reordering dims
            with self.assertRaises(Exception):
                result = fn(a, shape, (1, 0))

            # Adding outermost dimensions
            a = make_arg((5, 5))
            target_shape = (3, 3, 5, 5)
            result = fn(a, target_shape, (2, 3))

            self.assertEqual(result.shape, target_shape)
            self.assertEqual(a.broadcast_to(target_shape), result)

            # Expands
            a = make_arg((1, 5, 1))
            target_shape = (3, 5, 7)
            result = fn(a, target_shape, (0, 1, 2))

            self.assertEqual(result.shape, target_shape)
            self.assertEqual(a.expand_as(result), result)

            # Unsqueezes
            a = make_arg((1, 2, 3))
            target_shape = (1, 2, 1, 3)
            result = fn(a, target_shape, (0, 1, 3))

            self.assertEqual(result.shape, target_shape)
            self.assertEqual(a.unsqueeze(2), result)

            # Adds outermost, expands, and unsqueezes
            a = make_arg((1, 2, 3))
            target_shape = (4, 1, 7, 2, 3, 3)
            result = fn(a, target_shape, (1, 3, 4))

            self.assertEqual(result.shape, target_shape)
            a.unsqueeze_(3)
            a.unsqueeze_(1)
            a.unsqueeze_(0)
            self.assertEqual(a.expand_as(result), result)


instantiate_device_type_tests(TestPrims, globals())

if __name__ == "__main__":
    run_tests()
