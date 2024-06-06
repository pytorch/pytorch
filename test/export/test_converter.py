# Owner(s): ["oncall: export"]

import unittest
from typing import Dict, Tuple

import torch

import torch.utils._pytree as pytree

from torch._dynamo.test_case import TestCase
from torch._export.converter import TS2EPConverter
from torch.export import ExportedProgram
from torch.testing._internal.common_utils import run_tests

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


class TestConverter(TestCase):
    def _check_equal_ts_ep_converter(self, mod, inp) -> ExportedProgram:
        ts_model = torch.jit.script(mod)
        ep = TS2EPConverter(ts_model, inp).convert()
        ep_out, _ = pytree.tree_flatten(ep.module()(*inp))
        orig_out, _ = pytree.tree_flatten(mod(*inp))
        self.assertEqual(len(ep_out), len(orig_out))
        for ep_t, orig_t in zip(ep_out, orig_out):
            if isinstance(ep_t, torch.Tensor):
                self.assertEqual(ep_t.shape, orig_t.shape)
                self.assertTrue(torch.allclose(ep_t, orig_t))
            else:
                self.assertEqual(ep_t, orig_t)
        return ep

    def test_ts2ep_converter_basic(self):
        class MSingle(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        class MMulti(torch.nn.Module):
            def forward(self, x, y):
                x = x.cos() + 1
                y = y.sin() - 1
                return x, y

        inp = (torch.ones(1, 3), torch.ones(1, 3))
        self._check_equal_ts_ep_converter(MSingle(), inp)
        self._check_equal_ts_ep_converter(MMulti(), inp)

    def test_ts2ep_converter_container_output(self):
        # Output is a List.
        class MOutputList(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return [a, b]

        # Output is a Tuple.
        class MOutputTuple(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return (a, b)

        # Output is a Dict.
        class MOutputDict(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                a = x * x
                b = y + y
                return {"data": {"mul": a, "add": b}}

        inp = (torch.tensor(4), torch.tensor(4))

        self._check_equal_ts_ep_converter(MOutputList(), inp)
        self._check_equal_ts_ep_converter(MOutputTuple(), inp)
        self._check_equal_ts_ep_converter(MOutputDict(), inp)

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

    def test_aten___getitem___list(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = torch.split(x, 2)
                return y[0]

        inp = (torch.rand((3, 2)),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten___getitem___dict(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = torch.split(x, 2)
                d_int = {0: y[0], 1: y[1]}
                d_str = {"0": y[0], "1": y[1]}
                d_bool = {True: y[0], False: y[1]}
                d_float = {0.1: y[0], 2.3: y[1]}
                return d_int[0], d_str["0"], d_bool[True], d_float[0.1]

        inp = (torch.rand((3, 2)),)
        self._check_equal_ts_ep_converter(Module(), inp)

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

    def test_convert_if_basic(self):
        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor):
                if x:
                    return y * y
                else:
                    return y + y

        inp = (torch.tensor(True), torch.tensor(4))
        ep = self._check_equal_ts_ep_converter(M(), inp)

        torch.testing.assert_close(
            ep.module()(torch.tensor(False), torch.tensor(4)),
            M()(torch.tensor(False), torch.tensor(4)),
        )

    def test_convert_if_multiple_out(self):
        class M(torch.nn.Module):
            def true_fn(self, y, z):
                return (z * z, z + z)

            def false_fn(self, y, z):
                return (y * y * y, y + y)

            def forward(self, x: torch.Tensor, y: torch.Tensor):
                z = y * y

                if x:
                    res = self.true_fn(y, z)
                else:
                    res = self.false_fn(y, z)

                return res[0] + res[1]

        inp = (torch.tensor(True), torch.tensor(4))
        ep = self._check_equal_ts_ep_converter(M(), inp)

        torch.testing.assert_close(
            ep.module()(torch.tensor(False), torch.tensor(4)),
            M()(torch.tensor(False), torch.tensor(4)),
        )

    def test_profiler__record_function(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                handle = torch.ops.profiler._record_function_enter_new("foo", None)
                y = x * 2 + 4
                torch.ops.profiler._record_function_exit(handle)
                return y

        x = torch.randn(10, 10)
        self._check_equal_ts_ep_converter(Module(), (x,))

    def test_aten_floordiv(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x // 2

        x = torch.randn(10, 10)
        self._check_equal_ts_ep_converter(Module(), (x,))

    def test_aten___is__(self):
        class Module(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                z = x + 1
                return x is y, z

        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten___isnot__(self):
        class Module(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                z = x + 1
                return x is not y, z

        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten___not__(self):
        class Module(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                z = x + 1
                return not (x is not y), z

        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_ts2ep_converter_unpack(self):
        class MUnpackList(torch.nn.Module):
            def forward(self, x):
                x, y = torch.split(x, 2)
                return x + y

        class MUnpackTuple(torch.nn.Module):
            def forward(self, x_tuple: Tuple[torch.Tensor, torch.Tensor]):
                x, y = x_tuple
                x = x.cos()
                return x + y

        inp = torch.ones(1, 4)
        self._check_equal_ts_ep_converter(MUnpackList(), inp)
        inp = ((torch.zeros(1, 4), torch.ones(1, 4)),)
        self._check_equal_ts_ep_converter(MUnpackTuple(), inp)

    def test_ts2ep_converter_contains(self):
        class MIn(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.dtype in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        class MNotIn(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.dtype in [-1]

        class MTensorIn(torch.nn.Module):
            def forward(self, x: torch.Tensor, x_dict: Dict[torch.Tensor, str]):
                return x in x_dict

        inp = (torch.tensor(4),)
        self._check_equal_ts_ep_converter(MIn(), inp)
        self._check_equal_ts_ep_converter(MNotIn(), inp)

        inp = (torch.tensor(4), {torch.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp)
        inp = (torch.tensor(1), {torch.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp)


if __name__ == "__main__":
    run_tests()
