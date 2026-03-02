import unittest

import torch
from torch.testing._internal.common_utils import IS_ARM64, TestCase, run_tests
from torch.utils.cpp_extension import load_inline


@unittest.skipUnless(IS_ARM64, "AArch64-only regression")
class TestVecBF16Transpose(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cpp_source = r"""
#include <ATen/cpu/vec/vec.h>
#include <torch/extension.h>

torch::Tensor bf16_transpose(torch::Tensor input) {
  TORCH_CHECK(input.dim() == 2, "expected a 2D tensor");
  TORCH_CHECK(
      input.scalar_type() == c10::ScalarType::BFloat16,
      "expected bfloat16 input");

  auto contiguous = input.contiguous();
  const auto m = contiguous.size(0);
  const auto n = contiguous.size(1);

  auto output = torch::empty({n, m}, contiguous.options());
  const auto* src =
      reinterpret_cast<const c10::BFloat16*>(contiguous.data_ptr<c10::BFloat16>());
  auto* dst =
      reinterpret_cast<c10::BFloat16*>(output.data_ptr<c10::BFloat16>());

  at::vec::transpose_mxn<c10::BFloat16>(src, n, dst, m, m, n);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bf16_transpose", &bf16_transpose);
}
"""

        cls._ext = load_inline(
            name="bf16_vec_transpose_ext",
            cpp_sources=cpp_source,
            functions=None,
            extra_cflags=["-std=c++17"],
            verbose=False,
        )

    def test_tile_shapes(self):
        shapes = [(8, 8), (7, 5), (5, 7), (3, 8), (8, 3)]
        for m, n in shapes:
            x = torch.randn(m, n, dtype=torch.float32)
            bf16 = x.to(torch.bfloat16)
            actual = self._ext.bf16_transpose(bf16)
            expected = bf16.float().transpose(0, 1).to(torch.bfloat16)
            self.assertEqual(actual, expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
