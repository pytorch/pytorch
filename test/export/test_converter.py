# Owner(s): ["oncall: export"]

import torch
from torch._dynamo.test_case import TestCase
from torch._export.converter import TS2EPConverter

from torch.testing._internal.common_utils import run_tests


class TestConverter(TestCase):
    def _check_equal_ts_ep_converter(self, mod_func, inp):
        mod = mod_func()
        ts_model = torch.jit.script(mod)
        print(ts_model.graph)
        ep = TS2EPConverter(ts_model, inp).convert()
        print(ep.graph)
        torch.testing.assert_close(ep.module()(*inp)[0], mod(*inp))
        print(ep.module()(*inp)[0])

    def test_ts2ep_converter_basic(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inp = (torch.ones(1, 3), torch.ones(1, 3))
        self._check_equal_ts_ep_converter(M, inp)

    def test_ts2ep_converter_list(self):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return [a, b]

        inp = (torch.tensor(4), torch.tensor(4))
        self._check_equal_ts_ep_converter(M, inp)

    def test_ts2ep_converter_tuple(self):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return (a, b)

        inp = (torch.tensor(4), torch.tensor(4))
        self._check_equal_ts_ep_converter(M, inp)

    def test_ts2ep_converter_dict(self):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return {"data": {"mul": a, "add": b}}

        inp = (torch.tensor(4), torch.tensor(4))
        self._check_equal_ts_ep_converter(M, inp)


if __name__ == "__main__":
    run_tests()
