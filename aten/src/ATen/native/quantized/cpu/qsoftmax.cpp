#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace native {

namespace {

Tensor qsoftmax(
    const Tensor& qx,
    const int64_t dim,
    const double output_scale,
    const int64_t output_zero_point) {
  Tensor rx = at::dequantize(qx);
  Tensor ry = at::softmax(rx, dim);
  return at::quantize_per_tensor(
      ry, output_scale, output_zero_point, qx.scalar_type());
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::softmax"), TORCH_FN(qsoftmax));
}

} // namespace

} // namespace native
} // namespace at
