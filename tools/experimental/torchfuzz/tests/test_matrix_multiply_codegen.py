import unittest

from torchfuzz.operators.matrix_multiply import (
    AddmmOperator,
    BmmOperator,
    MatmulOperator,
    MMOperator,
)
from torchfuzz.tensor_fuzzer import TensorSpec

import torch


class MatrixMultiplyCodegenTest(unittest.TestCase):
    """The matmul family must not emit a cast it has already made redundant.

    `fuzz_inputs_specs` gives every input the output dtype (see
    `_get_compatible_dtype`), so a `.to(output dtype)` in `codegen` converts
    nothing. It is not harmless: it puts two extra `_to_copy` nodes into every
    traced graph, and it means no generated program ever exercises a bare
    matmul, so a divergence cannot be attributed to the matmul rather than to
    the copy beside it.
    """

    def _spec(self, size: list[int], dtype: torch.dtype = torch.float32) -> TensorSpec:
        stride = [1]
        for dim in reversed(size[1:]):
            stride.append(stride[-1] * dim)
        return TensorSpec(size=tuple(size), stride=tuple(reversed(stride)), dtype=dtype)

    def test_matmul_emits_no_cast(self) -> None:
        code = MatmulOperator().codegen("out", ["a", "b"], self._spec([4, 8]))

        self.assertEqual("out = torch.matmul(a, b)", code)

    def test_mm_emits_no_cast(self) -> None:
        code = MMOperator().codegen("out", ["a", "b"], self._spec([4, 8]))

        self.assertEqual("out = torch.mm(a, b)", code)

    def test_bmm_emits_no_cast(self) -> None:
        code = BmmOperator().codegen("out", ["a", "b"], self._spec([2, 4, 8]))

        self.assertEqual("out = torch.bmm(a, b)", code)

    def test_addmm_emits_no_cast(self) -> None:
        code = AddmmOperator().codegen("out", ["a", "b", "c"], self._spec([4, 8]))

        self.assertEqual("out = torch.addmm(a, b, c)", code)

    def test_no_cast_at_any_float_dtype(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            code = MatmulOperator().codegen(
                "out", ["a", "b"], self._spec([4, 8], dtype)
            )

            self.assertNotIn(".to(", code, f"cast emitted at {dtype}")

    def test_inputs_are_specified_at_the_output_dtype(self) -> None:
        # This is what makes dropping the cast safe. If it ever stops holding,
        # codegen has to start casting again.
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            specs = MatmulOperator().fuzz_inputs_specs(self._spec([4, 8], dtype))

            for spec in specs:
                self.assertEqual(dtype, spec.dtype)

    def test_input_count_is_still_validated(self) -> None:
        with self.assertRaises(ValueError):
            MatmulOperator().codegen("out", ["a"], self._spec([4, 8]))
        with self.assertRaises(ValueError):
            AddmmOperator().codegen("out", ["a", "b"], self._spec([4, 8]))


if __name__ == "__main__":
    unittest.main()
