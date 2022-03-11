#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace native {

namespace {

inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.scalar_type() == c10::kQInt8 || qa.scalar_type() == c10::kQUInt8,
      "MatMul operands should use QInt8 or QUInt8 data types.");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "MatMul operands should have same data type.");
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine || qa.qscheme() == kPerTensorSymmetric,
      "Only per-tensor quantization is suported in Matmul.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Matmul must have the same quantization scheme.");
}

Tensor qmatmul(
    const Tensor& qa,
    const Tensor& qb,
    const double output_scale,
    const int64_t output_zero_point) {
  check_inputs(qa, qb);
  Tensor ra = at::dequantize(qa);
  Tensor rb = at::dequantize(qb);
  Tensor rc = at::matmul(ra, rb);
  return at::quantize_per_tensor(
      rc, output_scale, output_zero_point, qa.scalar_type());
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::matmul"), TORCH_FN(qmatmul));
}

} // namespace

} // namespace native
} // namespace at
