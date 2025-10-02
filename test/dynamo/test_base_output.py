# Owner(s): ["module: dynamo"]
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same


try:
    from diffusers.models import unet_2d
except ImportError:
    unet_2d = None


def maybe_skip(fn):
    if unet_2d is None:
        return unittest.skip("requires diffusers")(fn)
    return fn


class TestBaseOutput(torch._dynamo.test_case.TestCase):
    @maybe_skip
    def test_create(self):
        def fn(a):
            tmp = unet_2d.UNet2DOutput(a + 1)
            return tmp

        torch._dynamo.testing.standard_test(self, fn=fn, nargs=1, expected_ops=1)

    @maybe_skip
    def test_assign(self):
        def fn(a):
            tmp = unet_2d.UNet2DOutput(a + 1)
            tmp.sample = a + 2
            return tmp

        args = [torch.randn(10)]
        obj1 = fn(*args)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1.sample, obj2.sample))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def _common(self, fn, op_count):
        args = [
            unet_2d.UNet2DOutput(
                sample=torch.randn(10),
            )
        ]
        obj1 = fn(*args)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(cnts)(fn)
        obj2 = opt_fn(*args)
        self.assertTrue(same(obj1, obj2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, op_count)

    @maybe_skip
    def test_getattr(self):
        def fn(obj: unet_2d.UNet2DOutput):
            x = obj.sample * 10
            return x

        self._common(fn, 1)

    @maybe_skip
    def test_getitem(self):
        def fn(obj: unet_2d.UNet2DOutput):
            x = obj["sample"] * 10
            return x

        self._common(fn, 1)

    @maybe_skip
    def test_tuple(self):
        def fn(obj: unet_2d.UNet2DOutput):
            a = obj.to_tuple()
            return a[0] * 10

        self._common(fn, 1)

    @maybe_skip
    def test_index(self):
        def fn(obj: unet_2d.UNet2DOutput):
            return obj[0] * 10

        self._common(fn, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
