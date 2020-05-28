#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

namespace at {
namespace native {

DEFINE_DISPATCH(qelu_stub);

Tensor& quantized_elu_out(Tensor& result, const Tensor& self, Scalar alpha,
    Scalar scale, Scalar input_scale) {
  qelu_stub(self.device().type(), self, alpha, scale, input_scale, result);
  return result;
}

Tensor& quantized_elu_(Tensor& self, Scalar alpha, Scalar scale,
    Scalar input_scale) {
  Tensor qy = at::_empty_affine_quantized(self.sizes(), self.options(),
      self.q_scale(), self.q_zero_point());
  qelu_stub(self.device().type(), self, alpha, scale, input_scale, qy);
  // This can be optimized in a later PR if necessary.
  self.copy_(qy);
  return self;
}

Tensor quantized_elu(
    const Tensor& qx, Scalar alpha, Scalar scale, Scalar input_scale) {
  Tensor qy = at::_empty_affine_quantized(qx.sizes(), qx.options(),
      qx.q_scale(), qx.q_zero_point());
  qelu_stub(qx.device().type(), qx, alpha, scale, input_scale, qy);
  return qy;
}

Tensor& quantized_celu_(Tensor& self, Scalar alpha) {
  double inv_alpha = 1. / alpha.to<double>();
  return quantized_elu_(self, alpha, Scalar(1.0), Scalar(inv_alpha));
}

Tensor quantized_celu(const Tensor& qx, Scalar alpha) {
  double inv_alpha = 1. / alpha.to<double>();
  return quantized_elu(qx, alpha, Scalar(1.0), Scalar(inv_alpha));
}

}}  // namespace at::native
