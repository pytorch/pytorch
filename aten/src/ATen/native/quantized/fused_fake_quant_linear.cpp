#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/quantized/fake_quant_affine.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>

namespace at {
namespace native {

at::Tensor fused_fake_quant_linear_cpu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& buffer_min,
    const at::Tensor& buffer_max,
    double running_min,
    double running_max,
    int64_t window_size) {
  // Calculate min_max of the inputs and weight tensot
  at::Tensor x_min, x_max, w_min, w_max;

  std::tie(x_min, x_max) = at::_aminmax(x);
  std::tie(w_min, w_max) = at::_aminmax(weight);

  // For now compute quantization parameters using min-max values
  float qmin = std::numeric_limits<uint8_t>::min();
  float qmax = std::numeric_limits<uint8_t>::max();
  fbgemm::TensorQuantizationParams x_qparams = fbgemm::ChooseQuantizationParams(
      x_min.item().toFloat(),
      x_max.item().toFloat(),
      qmin,
      qmax,
      false /* preserve sparsity */,
      false /* force power of two*/);

  fbgemm::TensorQuantizationParams w_qparams = fbgemm::ChooseQuantizationParams(
      w_min.item().toFloat(),
      w_max.item().toFloat(),
      qmin,
      qmax,
      false /* preserve sparsity */,
      false /* force power of two*/);

  // Call the fake_quantize kernel
  at::Tensor fake_quant_x = at::fake_quantize_per_tensor_affine(
      x, x_qparams.scale, x_qparams.zero_point, qmin, qmax);
  at::Tensor fake_quant_w = at::fake_quantize_per_tensor_affine(
      weight, w_qparams.scale, w_qparams.zero_point, qmin, qmax);

  at::Tensor out = at::addmm(bias, fake_quant_x, fake_quant_w);
  return out;
}
} // namespace native
} // namespace at
