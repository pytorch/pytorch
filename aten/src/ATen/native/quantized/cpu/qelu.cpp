#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

namespace at {
namespace native {

DEFINE_DISPATCH(qelu_stub);

Tensor quantized_elu(
    const Tensor& qx, double output_scale, int64_t output_zero_point, Scalar alpha, Scalar scale, Scalar input_scale) {
  Tensor qy = at::_empty_affine_quantized(qx.sizes(), qx.options(),
      output_scale, output_zero_point);
  qelu_stub(qx.device().type(), qx, alpha, qy);
  return qy;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl("elu", quantized_elu);
}

}}  // namespace at::native
