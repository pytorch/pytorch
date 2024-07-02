# Owner(s): ["oncall: export"]

import unittest
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import torch

import torch.utils._pytree as pytree

from torch._dynamo.test_case import TestCase
from torch._export.converter import TS2EPConverter
from torch.export import ExportedProgram
from torch.testing._internal.common_utils import run_tests

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


class TestConverter(TestCase):
    def _check_equal_ts_ep_converter(
        self,
        M,
        inp,
        option: Union[List[str]] = None,
        check_persistent=False,
        lifted_tensor_constants=None,
    ) -> ExportedProgram:
        # By default, it tests both jit.trace and jit.script.
        if option is None:
            option = ["trace", "script"]

        if check_persistent:
            num_iterations = 10
        else:
            num_iterations = 1

        ep_list = []
        for opt in option:
            if opt == "script":
                # Separate two models for testing non-functional effects
                if check_persistent:
                    original_ts_model = torch.jit.script(M())
                    ts_model = torch.jit.script(M())
                    eager_model = M()
                else:
                    original_ts_model = torch.jit.script(M)
                    ts_model = torch.jit.script(M)
                    eager_model = M
            elif opt == "trace":
                if check_persistent:
                    original_ts_model = torch.jit.trace(M(), inp)
                    ts_model = torch.jit.trace(M(), inp)
                    eager_model = M()
                else:
                    original_ts_model = torch.jit.trace(M, inp)
                    ts_model = torch.jit.trace(M, inp)
                    eager_model = M
            else:
                raise RuntimeError(f"Unrecognized mode for torch.jit: {opt}")

            ep = TS2EPConverter(ts_model, inp).convert()
            ep_list.append(ep)

            for _ in range(num_iterations):
                orig_out, _ = pytree.tree_flatten(original_ts_model(*inp))
                ep_out, _ = pytree.tree_flatten(ep.module()(*inp))

                # Check module.
                if isinstance(eager_model, torch.nn.Module):
                    expected_state_dict = OrderedDict()
                    expected_state_dict.update(ts_model.state_dict())
                    if lifted_tensor_constants:
                        expected_state_dict.update(lifted_tensor_constants)
                    self.assertEqual(
                        ep.state_dict.keys(),
                        expected_state_dict.keys(),
                    )

                # Check results
                self._check_tensor_list_equal(ep_out, orig_out)
        return ep_list

    def _check_tensor_list_equal(self, xs: List[torch.Tensor], ys: List[torch.Tensor]):
        self.assertEqual(len(xs), len(ys))
        for x, y in zip(xs, ys):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                self.assertEqual(x.shape, y.shape)
                self.assertTrue(torch.allclose(x, y))
            else:
                self.assertEqual(type(x), type(y))
                self.assertEqual(x, y)

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

        # Traced function must use immutable structure as output.
        self._check_equal_ts_ep_converter(MOutputList(), inp, ["script"])
        self._check_equal_ts_ep_converter(MOutputTuple(), inp)
        self._check_equal_ts_ep_converter(MOutputDict(), inp, ["script"])

    def test_aten_dim(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                num_dim = x.dim()
                return torch.ones(num_dim)

        inp = (torch.ones(1, 3),)
        self._check_equal_ts_ep_converter(Module(), inp)

    def test_aten_len(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                length = len(x)
                return torch.ones(length)

        # aten::len.Tensor
        inp = (torch.ones(2, 3),)
        self._check_equal_ts_ep_converter(Module(), inp)

        class Module(torch.nn.Module):
            def forward(self, x: List[int]):
                length = len(x)
                return torch.ones(length)

        # aten::len.t
        inp = ([1, 2, 3],)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            def forward(self, x: Dict[int, str]):
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_int
        inp = ({1: "a", 2: "b", 3: "c"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            def forward(self, x: Dict[bool, str]):
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_bool
        inp = ({True: "a", False: "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            def forward(self, x: Dict[float, str]):
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_float
        inp = ({1.2: "a", 3.4: "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            def forward(self, x: Dict[torch.Tensor, str]):
                length = len(x)
                return torch.ones(length)

        # aten::len.Dict_Tensor
        inp = ({torch.zeros(2, 3): "a", torch.ones(2, 3): "b"},)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        # aten::len.str and aten::len.Dict_str are not supported
        # since torch._C._jit_flatten does not support str
        # inp = ("abcdefg",)
        # self._check_equal_ts_ep_converter(Module(), inp)
        # inp = ({"a": 1, "b": 2},)
        # self._check_equal_ts_ep_converter(Module(), inp)

    def test_prim_min(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x_len = len(x)
                y_len = len(y)

                # prim::min.int
                len_int = min(x_len, y_len)

                # prim::min.float
                len_float = int(min(x_len * 2.0, y_len * 2.0))

                # prim::min.self_int
                len_self_int = min([x_len, y_len])

                # prim::min.self_float
                len_self_float = int(min([x_len * 2.0, y_len * 2.0]))

                # prim::min.float_int
                len_float_int = int(min(x_len * 2.0, y_len))

                # prim::min.int_float
                len_int_float = int(min(x_len, y_len * 2.0))

                return torch.ones(
                    len_int
                    + len_float
                    + len_self_int
                    + len_self_float
                    + len_float_int
                    + len_int_float
                )

        inp = (torch.randn(10, 2), torch.randn(5))
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
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
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
        ep_list = self._check_equal_ts_ep_converter(M(), inp)

        for ep in ep_list[1:]:
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

        # Traced function must return output that has tensors.
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_aten___isnot__(self):
        class Module(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                z = x + 1
                return x is not y, z

        # Traced function must return output that has tensors.
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_aten___not__(self):
        class Module(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[bool, torch.Tensor]:
                z = x + 1
                return not (x is not y), z

        # Traced function must return output that has tensors.
        inp = (torch.randn(10, 10), torch.rand(10, 10))
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

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
        self._check_equal_ts_ep_converter(orig_m, inp)

        orig_m = SuperNestedM(3)
        self._check_equal_ts_ep_converter(orig_m, inp)

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
        self._check_equal_ts_ep_converter(orig_m, inp)
        orig_m = SuperNestedM()
        self._check_equal_ts_ep_converter(orig_m, inp)

    def test_convert_nn_module_with_nested_if_and_buffer(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("w", torch.randn(1))
                self.count = 1

            def forward(self, x: torch.Tensor):
                return self.w + x + self.count

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
        # TODO: fix trace: state_dict is not equal.
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp, ["script"])

        t = inp[0]
        t -= 1
        for ep in ep_list:
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
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp)

        t = inp[0]
        t -= 0.8
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Nested module testing.
        inp = (torch.ones(3),)
        orig_m = NestedM(3)
        # TODO: fix trace: state_dict is not equal.
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp, ["script"])

        t = inp[0]
        t -= 0.8
        for ep in ep_list:
            torch.testing.assert_close(
                ep.module()(*inp),
                orig_m(*inp),
            )

        # Super nested module testing.
        inp = (torch.ones(3),)
        orig_m = SuperNestedM1(3)
        # TODO: fix trace: state_dict is not equal.
        ep_list = self._check_equal_ts_ep_converter(orig_m, inp, ["script"])

        t = inp[0]
        t -= 0.8
        for ep in ep_list:
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

        # Traced function must return output that has tensors.
        inp = (torch.tensor(4),)
        self._check_equal_ts_ep_converter(MIn(), inp, ["script"])
        self._check_equal_ts_ep_converter(MNotIn(), inp, ["script"])

        # TODO: update test to use reference for in.
        inp = (torch.tensor(4), {torch.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])
        inp = (torch.tensor(1), {torch.tensor(4): "foo"})
        self._check_equal_ts_ep_converter(MTensorIn(), inp, ["script"])

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

        ep_list = self._check_equal_ts_ep_converter(func2, inp)

        t = inp[0]
        t -= 1
        for ep in ep_list[1:]:
            torch.testing.assert_close(
                ep.module()(*inp),
                func2(*inp),
            )

    def test_implicit_constant_to_tensor_handling(self):
        def func1(x):
            return x + 2

        def func2(x, y):
            return x * y / (x - 2 * y) + y

        def func3(x):
            return x + torch.tensor([3])

        def func4():
            val = torch.tensor(float("inf"))
            return torch.full((10, 10), val)

        def func5():
            x = -1
            return x * torch.ones(1, dtype=torch.float), torch.zeros(
                1, dtype=torch.float
            )

        def func6(x1, x2, x3, x4):
            return (
                x1.numel(),
                x1.size(),
                x2.numel(),
                x2.size(),
                x3.numel(),
                x3.size(),
                x4.numel(),
                x4.size(),
                torch.ones(x1.numel()),  # Just make sure downstream ops still work.
                torch.ones(x1.size()),  # Just make sure downstream ops still work.
            )

        class M1(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.x = torch.tensor(value)

            def forward(self):
                return self.x.clone()

        class M2(torch.nn.Module):
            def forward(self, x):
                return torch.tensor(4) + x

        inp = (torch.randn([2, 2]),)
        self._check_equal_ts_ep_converter(func1, inp)
        inp = (torch.randn([2, 2]), torch.randn([2, 2]))
        self._check_equal_ts_ep_converter(func2, inp)

        inp = (torch.randn([2, 2]),)
        self._check_equal_ts_ep_converter(func3, inp)

        self._check_equal_ts_ep_converter(func4, ())
        self._check_equal_ts_ep_converter(M1(5), ())

        inp = (torch.randn(2),)
        self._check_equal_ts_ep_converter(M2(), inp)

        self._check_equal_ts_ep_converter(func5, ())
        inp = (
            torch.randn([2, 3, 4]).to(torch.int8),
            torch.randn([2, 3, 4]).to(torch.int32),
            torch.randn([2, 3, 4]).to(torch.float32),
            torch.randn([2, 3, 4]).to(torch.float64),
        )
        self._check_equal_ts_ep_converter(func6, inp)

    def test_prim_tolist(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> List[int]:
                return x.tolist()

        inp = (torch.tensor([1, 2, 3]),)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> List[List[int]]:
                return x.tolist()

        inp = (torch.tensor([[1, 2, 3], [4, 5, 6]]),)
        self._check_equal_ts_ep_converter(Module(), inp, ["script"])

    def test_get_tensor_constants(self):
        # Since self.data is only read but not written, it is lifted as
        # constant tensors.
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.data = torch.randn(3, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.data

        class Goo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.data = torch.randn(3, 2)
                self.foo = Foo()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.data + self.foo.data + self.foo(x)

        inp = (torch.randn(3, 2),)
        goo = Goo()
        self._check_equal_ts_ep_converter(goo, inp)

    def test_prim_SetAttr(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("data", torch.ones(3, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.data = self.data + x
                return x + x

        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module, inp, ["script"], check_persistent=True
        )

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("data", torch.ones(3, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.data = self.data + x
                return x + self.data

        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module, inp, ["script"], check_persistent=True
        )

        # export lifts a tensor constant (self.data) as an input if it is not assigned.
        # If it is assigned, export will error and ask users to register it as a buffer.
        # In converter, we change tensor constants that are assigned as a buffer automatically,
        # since it might be hard to manually register them as buffers.
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.data = torch.ones(3, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.data = self.data + x
                return x + self.data

        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module,
            inp,
            ["script"],
            check_persistent=True,
            lifted_tensor_constants=OrderedDict([("data", torch.ones(3, 2))]),
        )

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.count = 0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.count += 1
                return x + self.count

        # check_persistent is False since export specializes on non-tensor constants
        inp = (torch.ones(3, 2),)
        self._check_equal_ts_ep_converter(
            Module(), inp, ["script"], check_persistent=False
        )

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.count = 0

            def forward(self, x):
                count1 = self.count
                self.count += 1
                count2 = self.count
                self.count += 1
                count3 = self.count
                return x + count1 + count2 + count3

        inp = (torch.ones(1),)
        self._check_equal_ts_ep_converter(M(), inp, ["script"], check_persistent=False)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("w2", torch.ones(1))

            def forward(self, x: torch.Tensor):
                self.w2 += 1
                return self.w2

        inp = (torch.ones(1),)
        self._check_equal_ts_ep_converter(M, inp, ["script"], check_persistent=True)


if __name__ == "__main__":
    run_tests()
