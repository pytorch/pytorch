# Owner(s): ["oncall: export"]

import torch
from torch._dynamo.test_case import TestCase
from torch._export.converter import TS2EPConverter

import torch.utils._pytree as pytree

from torch.testing._internal.common_utils import run_tests


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

    def test_aten_dim(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                num_dim = x.dim()
                return torch.ones(num_dim)

        inp = (torch.ones(1, 3),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten_len(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                length = len(x)
                return torch.ones(length)

        inp = (torch.ones(2, 3),)
        self._check_equal_ts_ep_converter(Module(), inp)


if __name__ == "__main__":
    run_tests()
