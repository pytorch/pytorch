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

        # Check module.
        if isinstance(mod, torch.nn.Module):
            self.assertEqual(
                ep.module().state_dict().keys(),
                mod.state_dict().keys(),
            )

        # Check results.
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

        inp = (torch.ones(4),)
        self._check_equal_ts_ep_converter(MUnpackList(), inp)
        inp = ((torch.zeros(1, 4), torch.ones(1, 4)),)
        self._check_equal_ts_ep_converter(MUnpackTuple(), inp)

    def test_convert_nn_module_with_nested_param(self):
        class M(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                return self.linear(x)

        class NestedM(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)
                self.m = M(dim)

            def forward(self, x: torch.Tensor):
                return self.linear(self.m(x))

        class SuperNestedM(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)
                self.m = NestedM(dim)

            def forward(self, x: torch.Tensor):
                return self.linear(self.m(x))

        inp = (torch.ones(3),)
        orig_m = NestedM(3)
        ep = self._check_equal_ts_ep_converter(orig_m, inp)
        orig_m = SuperNestedM(3)
        ep = self._check_equal_ts_ep_converter(orig_m, inp)

    def test_convert_nn_module_with_nested_buffer(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("w", torch.randn(1))

            def forward(self, x: torch.Tensor):
                return self.w + x

        class NestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = M()
                self.register_buffer("w", torch.randn(1))

            def forward(self, x: torch.Tensor):
                return self.w + self.m(x)

        class SuperNestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = NestedM()
                self.register_buffer("w", torch.randn(1))

            def forward(self, x: torch.Tensor):
                return self.w + self.m(x)

        inp = (torch.ones(1),)
        orig_m = NestedM()
        ep = self._check_equal_ts_ep_converter(orig_m, inp)
        orig_m = SuperNestedM()
        ep = self._check_equal_ts_ep_converter(orig_m, inp)

    def test_convert_nn_module_with_nested_if_and_buffer(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("w", torch.randn(1))

            def forward(self, x: torch.Tensor):
                return self.w + x

        class NestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = M()
                self.m2 = M()
                self.register_buffer("w", torch.randn(1))

            def forward(self, x: torch.Tensor):
                if torch.sum(x) > 1:
                    return self.w + self.m1(x)
                else:
                    return self.w + self.m2(x)

        # Super nested, parameters neeed to lifted
        # multiple times.
        class SuperNestedM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = NestedM()
                self.m2 = NestedM()
                self.register_buffer("w", torch.randn(1))

            def forward(self, x: torch.Tensor):
                if torch.max(x) > 1:
                    return self.w + self.m1(x)
                else:
                    return self.w + self.m2(x)

        # Super nested module testing.
        inp = (torch.ones(1),)
        orig_m = SuperNestedM()
        ep = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 1
        torch.testing.assert_close(
            ep.module()(*inp),
            orig_m(*inp),
        )

    def test_convert_nn_module_with_nested_if_and_param(self):
        class M(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                return self.linear(x)

        class NestedM(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.m1 = M(dim)
                self.m2 = M(dim)
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                if torch.sum(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Super nested, parameters neeed to lifted
        # multiple times.
        class SuperNestedM1(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.m1 = NestedM(dim)
                self.m2 = NestedM(dim)
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                if torch.max(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Super nested, even the input needs to be
        # lifted recursively due to value propogation optimiztaion.
        class SuperNestedM2(torch.nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.m1 = NestedM(dim)
                self.m2 = NestedM(dim)
                self.linear = torch.nn.Linear(dim, dim)

            def forward(self, x: torch.Tensor):
                if torch.sum(x) > 1:
                    return self.linear(self.m1(x))
                else:
                    return self.linear(self.m2(x))

        # Basic module testing.
        inp = (torch.ones(3),)
        orig_m = M(3)
        ep = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        torch.testing.assert_close(
            ep.module()(*inp),
            orig_m(*inp),
        )

        # Nested module testing.
        inp = (torch.ones(3),)
        orig_m = NestedM(3)
        ep = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        torch.testing.assert_close(
            ep.module()(*inp),
            orig_m(*inp),
        )

        # Super nested module testing.
        inp = (torch.ones(3),)
        orig_m = SuperNestedM1(3)
        ep = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        torch.testing.assert_close(
            ep.module()(*inp),
            orig_m(*inp),
        )

        # # Super nested module testing.
        # inp = (torch.ones(3),)
        # orig_m = SuperNestedM2(3)
        # ep = self._check_equal_ts_ep_converter(orig_m, inp)

        # t = inp[0]
        # t -= 0.8
        # torch.testing.assert_close(
        #     ep.module()(*inp),
        #     orig_m(*inp),
        # )

    def test_ts2ep_converter_contains(self):
        class MIn(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.dtype in [torch.float32, torch.float64]

        class MNotIn(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.dtype in [torch.int8]

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

    def test_ts2ep_converter_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            torch.library.define(
                "mylib::foo",
                "(Tensor x) -> Tensor",
                lib=lib,
            )

            # PyTorch custorm op implementation
            @torch.library.impl(
                "mylib::foo",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def foo_impl(x):
                return x + x

            # Meta function of the custom op.
            @torch.library.impl_abstract(
                "mylib::foo",
                lib=lib,
            )
            def foo_meta(x):
                return x + x

            class M(torch.nn.Module):
                def forward(self, x):
                    return torch.ops.mylib.foo(x)

            inp = (torch.randn(3, 3),)
            m = M()
            self._check_equal_ts_ep_converter(m, inp)

    def test_convert_func_without_param(self):
        def func1(x, y):
            return x + y

        def func2(x, y):
            if x.sum() > 0:
                return x + y
            else:
                return x - y

        inp = (
            torch.tensor(1),
            torch.tensor(1),
        )
        self._check_equal_ts_ep_converter(func1, inp)

        ep = self._check_equal_ts_ep_converter(func2, inp)

        t = inp[0]
        t -= 1
        torch.testing.assert_close(
            ep.module()(*inp),
            func2(*inp),
        )


if __name__ == "__main__":
    run_tests()
