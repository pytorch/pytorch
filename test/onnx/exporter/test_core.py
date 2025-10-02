# Owner(s): ["module: onnx"]
"""Unit tests for the _core module."""

from __future__ import annotations

import ml_dtypes
import numpy as np

import torch
from torch.onnx._internal.exporter import _core
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class TorchTensorTest(common_utils.TestCase):
    @common_utils.parametrize(
        "dtype, np_dtype",
        [
            (torch.bfloat16, ml_dtypes.bfloat16),
            (torch.bool, np.bool_),
            (torch.complex128, np.complex128),
            (torch.complex64, np.complex64),
            (torch.float16, np.float16),
            (torch.float32, np.float32),
            (torch.float64, np.float64),
            (torch.float8_e4m3fn, ml_dtypes.float8_e4m3fn),
            (torch.float8_e4m3fnuz, ml_dtypes.float8_e4m3fnuz),
            (torch.float8_e5m2, ml_dtypes.float8_e5m2),
            (torch.float8_e5m2fnuz, ml_dtypes.float8_e5m2fnuz),
            (torch.int16, np.int16),
            (torch.int32, np.int32),
            (torch.int64, np.int64),
            (torch.int8, np.int8),
            (torch.uint16, np.uint16),
            (torch.uint32, np.uint32),
            (torch.uint64, np.uint64),
            (torch.uint8, np.uint8),
            (torch.float4_e2m1fn_x2, ml_dtypes.float4_e2m1fn),
        ],
    )
    def test_numpy_returns_correct_dtype(self, dtype: torch.dtype, np_dtype):
        if dtype == torch.float4_e2m1fn_x2:
            tensor = _core.TorchTensor(torch.tensor([1], dtype=torch.uint8).view(dtype))
        else:
            tensor = _core.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.numpy().dtype, np_dtype)
        self.assertEqual(tensor.__array__().dtype, np_dtype)
        self.assertEqual(np.array(tensor).dtype, np_dtype)

    @common_utils.parametrize(
        "dtype",
        [
            torch.bfloat16,
            torch.bool,
            torch.complex128,
            torch.complex64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.int8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.uint8,
        ],
    )
    def test_tobytes(self, dtype: torch.dtype):
        tensor = _core.TorchTensor(torch.tensor([1], dtype=dtype))
        self.assertEqual(tensor.tobytes(), tensor.numpy().tobytes())

    def test_tobytes_float4(self):
        tensor = _core.TorchTensor(
            torch.tensor([1], dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        )
        self.assertEqual(tensor.tobytes(), b"\x01")


if __name__ == "__main__":
    common_utils.run_tests()
