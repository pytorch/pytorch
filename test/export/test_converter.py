# Owner(s): ["oncall: export"]

import unittest

import torch

import torch.utils._pytree as pytree

from torch._dynamo.test_case import TestCase
from torch._export.converter import TS2EPConverter

from torch.testing._internal.common_utils import run_tests

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


class TestConverter(TestCase):
    def _check_equal_ts_ep_converter(self, mod, inp):
        ts_model = torch.jit.script(mod)
        ep = TS2EPConverter(ts_model, inp).convert()
        ep_out, _ = pytree.tree_flatten(ep.module()(*inp))
        orig_out, _ = pytree.tree_flatten(mod(*inp))
        self.assertEqual(len(ep_out), len(orig_out))
        for ep_t, orig_t in zip(ep_out, orig_out):
            self.assertEqual(ep_t.shape, orig_t.shape)
            self.assertTrue(torch.allclose(ep_t, orig_t))

    def test_ts2ep_converter_basic(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        m = Module()
        inp = (torch.ones(1, 3), torch.ones(1, 3))

        ts_model = torch.jit.script(m)
        ep = TS2EPConverter(ts_model, inp).convert()

        torch.testing.assert_close(ep.module()(*inp)[0], m(*inp))

    def test_prim_device(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                device = x.device
                return torch.ones(2, 3, device=device)

        inp = (torch.rand(3, 4),)
        self._check_equal_ts_ep_converter(Module(), inp)

    @requires_cuda
    def test_prim_device_cuda(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                device = x.device
                return torch.ones(2, 3, device=device)

        inp = (torch.rand((3, 4), device="cuda:0"),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_dtype(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                dtype = x.dtype
                return torch.ones(2, 3, dtype=dtype)

        for dtype in [
            torch.float32,
            torch.double,
        ]:
            inp = (torch.rand((3, 4), dtype=dtype),)
            self._check_equal_ts_ep_converter(Module(), inp)

        for dtype in [
            torch.uint8,
            torch.int8,
            torch.int32,
        ]:
            inp = (torch.randint(high=128, size=(3, 4), dtype=dtype),)
            self._check_equal_ts_ep_converter(Module(), inp)


if __name__ == "__main__":
    run_tests()
