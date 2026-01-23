# Owner(s): ["module: custom-operators"]

from dataclasses import dataclass

import torch
import torch.utils._pytree as pytree
from torch._library.infer_schema import infer_schema
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


@dataclass
class Point:
    x: torch.Tensor
    y: torch.Tensor


pytree.register_dataclass(Point)


class TestPytreeOps(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestPytreeOps", "FRAGMENT")  # noqa: TOR901
        super().setUp()

    def tearDown(self):
        self.lib._destroy()
        super().tearDown()

    def test_schema_inference_list_types_before_pytree(self):
        def fn_list_tensor(
            list_tensor: list[torch.Tensor],
            list_int: list[int],
            list_float: list[float],
            list_bool: list[bool],
            pytree_list: list,
        ) -> torch.Tensor:
            return list_tensor[0]

        schema = infer_schema(fn_list_tensor, mutates_args=())
        self.assertEqual(
            schema,
            "(Tensor[] list_tensor, SymInt[] list_int, float[] list_float, bool[] list_bool, builtins.list pytree_list) -> Tensor",
        )

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_dict_input(self):
        # Use define/impl API instead of custom_op
        torch.library.define(
            "_TestPytreeOps::dict_op",
            "(builtins.dict d, Tensor t) -> Tensor",
            lib=self.lib,
        )

        @torch.library.impl("_TestPytreeOps::dict_op", "CPU", lib=self.lib)
        def dict_op_impl(d: dict, t: torch.Tensor) -> torch.Tensor:
            return torch.sin(d["x"] - d["y"] + t)

        d = {"x": torch.randn(2, 3), "y": torch.randn(2, 3)}
        t = torch.randn(2, 3)
        y = torch.ops._TestPytreeOps.dict_op(d, t)
        self.assertEqual(y, torch.sin(d["x"] - d["y"] + t))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_list_input(self):
        @torch.library.custom_op("_TestPytreeOps::list_op", mutates_args=())
        def foo(lst: list, t: torch.Tensor) -> torch.Tensor:
            return torch.sin(lst[0] + lst[1] + t)

        lst = [torch.randn(2, 3), torch.randn(2, 3)]
        t = torch.randn(2, 3)
        y = torch.ops._TestPytreeOps.list_op(lst, t)
        self.assertEqual(y, torch.sin(lst[0] + lst[1] + t))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_dataclass_input(self):
        @torch.library.custom_op("_TestPytreeOps::dataclass_op", mutates_args=())
        def dataclass_op_impl(a: Point) -> torch.Tensor:
            return torch.sqrt(torch.sum((a.x - a.y) ** 2))

        x = Point(x=torch.randn(2, 3), y=torch.randn(2, 3))
        y = torch.ops._TestPytreeOps.dataclass_op(x)
        self.assertEqual(y, torch.sqrt(torch.sum((x.x - x.y) ** 2)))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_tuple_input(self):
        @torch.library.custom_op("_TestPytreeOps::tuple_op", mutates_args=())
        def foo(tup: tuple, t: torch.Tensor) -> torch.Tensor:
            return torch.cos(tup[0] * tup[1] + t)

        tup = (torch.randn(2, 3), torch.randn(2, 3))
        t = torch.randn(2, 3)
        y = torch.ops._TestPytreeOps.tuple_op(tup, t)
        self.assertEqual(y, torch.cos(tup[0] * tup[1] + t))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_nested_pytree_input(self):
        @torch.library.custom_op("_TestPytreeOps::nested_op", mutates_args=())
        def foo(d: dict) -> torch.Tensor:
            return torch.sin(d["a"][0] + d["b"]["x"])

        d = {
            "a": [torch.randn(2, 3), torch.randn(2, 3)],
            "b": {"x": torch.randn(2, 3), "y": torch.randn(2, 3)},
        }
        y = torch.ops._TestPytreeOps.nested_op(d)
        self.assertEqual(y, torch.sin(d["a"][0] + d["b"]["x"]))


if __name__ == "__main__":
    run_tests()
