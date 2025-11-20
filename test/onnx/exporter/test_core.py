# Owner(s): ["module: onnx"]
"""Unit tests for the _core module."""

from __future__ import annotations

import io
import os
import tempfile

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


class TorchTensorToFileTest(common_utils.TestCase):
    def _roundtrip_file(self, tensor: _core.TorchTensor) -> bytes:
        expected = tensor.tobytes()
        # NamedTemporaryFile (binary)
        with tempfile.NamedTemporaryFile() as tmp:
            tensor.tofile(tmp)
            tmp.seek(0)
            data = tmp.read()
        self.assertEqual(data, expected)

        # Explicit path write using open handle
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bin.dat")
            with open(path, "wb") as f:
                tensor.tofile(f)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), expected)

        return expected

    def test_tofile_basic_uint8(self):
        tensor = _core.TorchTensor(torch.arange(10, dtype=torch.uint8))
        self._roundtrip_file(tensor)

    def test_tofile_float32(self):
        tensor = _core.TorchTensor(
            torch.arange(0, 16, dtype=torch.float32).reshape(4, 4)
        )
        self._roundtrip_file(tensor)

    def test_tofile_bfloat16(self):
        tensor = _core.TorchTensor(torch.arange(0, 8, dtype=torch.bfloat16))
        self._roundtrip_file(tensor)

    def test_tofile_float4_packed(self):
        # 3 packed bytes -> 6 logical float4 values (when unpacked), but we want packed bytes
        raw = torch.tensor([0x12, 0x34, 0xAB], dtype=torch.uint8)
        tensor = _core.TorchTensor(raw.view(torch.float4_e2m1fn_x2))
        expected = self._roundtrip_file(tensor)
        self.assertEqual(expected, bytes([0x12, 0x34, 0xAB]))

    def test_tofile_file_like_no_fileno(self):
        tensor = _core.TorchTensor(torch.arange(0, 32, dtype=torch.uint8))
        buf = io.BytesIO()
        tensor.tofile(buf)
        self.assertEqual(buf.getvalue(), tensor.tobytes())

    def test_tofile_text_mode_error(self):
        tensor = _core.TorchTensor(torch.arange(0, 4, dtype=torch.uint8))
        with tempfile.NamedTemporaryFile(mode="w") as tmp_text:
            path = tmp_text.name
            with open(path, "w") as f_text:
                with self.assertRaises(TypeError):
                    tensor.tofile(f_text)

    def test_tofile_non_contiguous(self):
        base = torch.arange(0, 64, dtype=torch.int32).reshape(8, 8)
        sliced = base[:, ::2]  # Stride in last dim -> non-contiguous
        self.assertFalse(sliced.is_contiguous())
        tensor = _core.TorchTensor(sliced)
        # Ensure bytes correspond to the contiguous clone inside implementation
        expected_manual = sliced.contiguous().numpy().tobytes()
        with tempfile.NamedTemporaryFile() as tmp:
            tensor.tofile(tmp)
            tmp.seek(0)
            data = tmp.read()
        self.assertEqual(data, expected_manual)
        self.assertEqual(tensor.tobytes(), expected_manual)


if __name__ == "__main__":
    common_utils.run_tests()
