# Owner(s): ["module: pt2-dispatcher"]
from __future__ import annotations

import typing
from typing import List, Optional, Union

import torch
from torch import Tensor, types
from torch.testing._internal.common_utils import run_tests, TestCase

if typing.TYPE_CHECKING:
    from collections.abc import Sequence


mutates_args = {}


class TestInferSchemaWithAnnotation(TestCase):
    def test_tensor(self):
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.clone() + y

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x, Tensor y) -> Tensor")

    def test_native_types(self):
        def foo_op(x: int) -> int:
            return x

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt x) -> SymInt")

        def foo_op_2(x: bool) -> bool:
            return x

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(bool x) -> bool")

        def foo_op_3(x: str) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(str x) -> SymInt")

        def foo_op_4(x: float) -> float:
            return x

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(float x) -> float")

    def test_torch_types(self):
        def foo_op_1(x: torch.types.Number) -> torch.types.Number:
            return x

        result = torch.library.infer_schema(foo_op_1, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_2(x: torch.dtype) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(ScalarType x) -> SymInt")

        def foo_op_3(x: torch.device) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(Device x) -> SymInt")

    def test_type_variants(self):
        def foo_op_1(x: typing.Optional[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_1, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")

        def foo_op_2(x: typing.Sequence[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_3(x: typing.List[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_4(x: typing.Optional[typing.Sequence[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_5(x: typing.Optional[typing.List[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_5, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_6(x: typing.Union[int, float, bool]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_6, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_7(x: typing.Union[int, bool, float]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_7, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

    def test_no_library_prefix(self):
        def foo_op(x: Tensor) -> Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_2(x: Tensor) -> torch.Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_3(x: torch.Tensor) -> Tensor:
            return x.clone()

        result = torch.library.infer_schema(foo_op_3, mutates_args=mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        def foo_op_4(x: List[int]) -> types.Number:
            return x[0]

        result = torch.library.infer_schema(foo_op_4, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> Scalar")

        def foo_op_5(x: Optional[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_5, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")

        def foo_op_6(x: Sequence[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_6, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_7(x: List[int]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_7, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        def foo_op_8(x: Optional[Sequence[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_8, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_9(x: Optional[List[int]]) -> int:
            return 1

        result = torch.library.infer_schema(foo_op_9, mutates_args=mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        def foo_op_10(x: Union[int, float, bool]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_10, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        def foo_op_11(x: Union[int, bool, float]) -> types.Number:
            return x

        result = torch.library.infer_schema(foo_op_11, mutates_args=mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

    def test_unsupported_annotation(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation D. It is not a type.",
        ):

            def foo_op(x: D) -> Tensor:  # noqa: F821
                return torch.Tensor(x)

            torch.library.infer_schema(foo_op, mutates_args=mutates_args)

        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation E. It is not a type.",
        ):

            def foo_op_2(x: Tensor) -> E:  # noqa: F821
                return x

            torch.library.infer_schema(foo_op_2, mutates_args=mutates_args)


if __name__ == "__main__":
    run_tests()
