# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestNativeScalarCoercion(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "dtype,value,inplace",
        [(torch.float64, value, False) for value in (2.3, 1e20, -70000.0)]
        + [
            (dtype, 1.00000001, inplace)
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
            for inplace in (False, True)
        ],
    )
    def test_float_scalar_coercion_precision(
        self, dtype: torch.dtype, value: float, inplace: bool
    ) -> None:
        from torch._native.registry import _scalar_arg_coercer

        expected_dtype = torch.float64 if dtype is torch.float64 else torch.float32
        op = torch.ops.aten.add_.Tensor if inplace else torch.ops.aten.add.Tensor
        shape = (3,) if dtype is torch.float64 else ()
        args, result_dtype = _scalar_arg_coercer(op)(
            (torch.ones(shape, dtype=dtype), value)
        )
        self.assertEqual(args[1].dtype, expected_dtype)
        self.assertEqual(
            args[1].item(), torch.tensor(value, dtype=expected_dtype).item()
        )
        self.assertEqual(result_dtype, dtype)

    @parametrize("value", (7, 70000))
    def test_unsigned_scalar_wrap_preserves_the_value(self, value: int) -> None:
        from torch._native.registry import _weak_wrap

        out = _weak_wrap(value, [torch.uint16], torch.uint16)
        self.assertEqual(out.item(), value)
        self.assertEqual(out.dtype, torch.uint16 if value == 7 else torch.int64)

    @parametrize(
        "value",
        [1e100, -1e100, 3.4028235e38, -3.4028235e38]
        + [
            sign * ((1 << 62) + (1 << 38) + offset)
            for sign in (-1, 1)
            for offset in (-1, 0, 1)
        ],
    )
    def test_fp32_rounding_matches_direct_tensor_cast(self, value: int | float) -> None:
        from torch._native.utils.scalar import f32

        dtype = torch.int64 if isinstance(value, int) else torch.float64
        expected = torch.tensor(value, dtype=dtype).float().item()
        self.assertEqual(f32(value), expected)


instantiate_parametrized_tests(TestNativeScalarCoercion)

if __name__ == "__main__":
    run_tests()
