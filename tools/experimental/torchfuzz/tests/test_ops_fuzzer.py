import unittest
from typing import cast

from torchfuzz.operators.elementwise_math import DivideOperator
from torchfuzz.ops_fuzzer import _get_template_filtered_operators
from torchfuzz.tensor_fuzzer import TensorSpec

import torch


class DivideOperatorTest(unittest.TestCase):
    def _codegen(self, rounding_mode: str, dtype: torch.dtype) -> str:
        operator = DivideOperator()
        operator._rounding_mode = rounding_mode
        return operator.codegen(
            "result",
            ["numerator", "recursive_divisor"],
            TensorSpec(size=(2,), stride=(1,), dtype=dtype),
        )

    def _execute_codegen(self, rounding_mode: str, dtype: torch.dtype) -> torch.Tensor:
        namespace: dict[str, object] = {
            "torch": torch,
            "numerator": torch.tensor([5, -5, 7], dtype=dtype),
            "recursive_divisor": torch.sub(
                torch.tensor([5, 7, 5], dtype=dtype),
                torch.tensor([5, 5, 8], dtype=dtype),
            ),
        }
        exec(self._codegen(rounding_mode, dtype), namespace)
        return cast(torch.Tensor, namespace["result"])

    def test_integer_trunc_replaces_zero_divisors(self) -> None:
        self.assertTrue(
            torch.equal(
                self._execute_codegen("trunc", torch.int32),
                torch.tensor([5, -2, -2], dtype=torch.int32),
            )
        )

    def test_integer_floor_replaces_zero_divisors(self) -> None:
        self.assertTrue(
            torch.equal(
                self._execute_codegen("floor", torch.int64),
                torch.tensor([5, -3, -3], dtype=torch.int64),
            )
        )

    def test_float_rounding_mode_preserves_divisor(self) -> None:
        self.assertEqual(
            self._codegen("trunc", torch.float32),
            "result = torch.divide(numerator, recursive_divisor, "
            "rounding_mode='trunc')",
        )

    def test_default_mode_preserves_integer_divisor(self) -> None:
        self.assertEqual(
            self._codegen("default", torch.int32),
            "result = torch.divide(numerator, recursive_divisor)",
        )


class SupportedOpsFilteringTest(unittest.TestCase):
    def _torch_op_names(self, supported_op: str) -> set[str | None]:
        operators = _get_template_filtered_operators(
            template="default", supported_ops=[supported_op]
        )
        return {operator.torch_op_name for operator in operators.values()}

    def test_expand_does_not_match_exp(self) -> None:
        names = self._torch_op_names("torch.expand")

        self.assertIn("torch.expand", names)
        self.assertNotIn("torch.exp", names)

    def test_exp_does_not_match_expand(self) -> None:
        names = self._torch_op_names("torch.exp")

        self.assertIn("torch.exp", names)
        self.assertNotIn("torch.expand", names)

    def test_unsqueeze_does_not_match_squeeze(self) -> None:
        names = self._torch_op_names("torch.unsqueeze")

        self.assertIn("torch.unsqueeze", names)
        self.assertNotIn("torch.squeeze", names)

    def test_squeeze_does_not_match_unsqueeze(self) -> None:
        names = self._torch_op_names("torch.squeeze")

        self.assertIn("torch.squeeze", names)
        self.assertNotIn("torch.unsqueeze", names)

    def test_registry_key_requires_exact_match(self) -> None:
        names = self._torch_op_names("expand")

        self.assertIn("torch.expand", names)
        self.assertNotIn("torch.exp", names)

    def test_partial_registry_key_does_not_match(self) -> None:
        names = self._torch_op_names("expan")

        self.assertNotIn("torch.expand", names)
        self.assertNotIn("torch.exp", names)
