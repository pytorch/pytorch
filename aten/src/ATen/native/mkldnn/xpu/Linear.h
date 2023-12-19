#include "BlasImpl.h"

namespace at {
namespace AtenIpexTypeXPU {

struct LinearConverter {
  LinearConverter() {
    is_fused_ = false;
  }

  // linear with post-ops
  template <typename Func>
  Tensor& call(
      Tensor& result,
      const Tensor& input,
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      Func func) {
    xpu::oneDNN::Attr attr = func();
    Tensor _bias = bias.has_value() ? bias.value() : at::Tensor();
    Tensor _input =
        input.dim() <= 2 ? input : xpu::oneDNN::contiguous_if_needed(input);
    return impl::matmul_fusion_variants(
        result, _input, weight, false, attr, is_fused_, _bias);
  }

  Tensor& call(
      Tensor& result,
      const Tensor& input,
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      xpu::oneDNN::Attr attr) {
    Tensor _bias = bias.has_value() ? bias.value() : at::Tensor();
    Tensor _input =
        input.dim() <= 2 ? input : xpu::oneDNN::contiguous_if_needed(input);
    return impl::matmul_fusion_variants(
        result, _input, weight, /*trans*/ true, attr, is_fused_, _bias);
  }

  bool is_fused() {
    return is_fused_;
  }

  bool is_fused_;
};

Tensor dpcpp_linear(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias);

// // IPEX customer linear for weight prepack
// Tensor linear(
//     const Tensor& input,
//     const Tensor& weight,
//     const c10::optional<Tensor>& bias_opt);

// Tensor& linear_out(
//     const Tensor& input,
//     const Tensor& weight,
//     const c10::optional<Tensor>& bias_opt,
//     Tensor& output);

// Tensor linear_pow(
//     const Tensor& input,
//     const Tensor& weight,
//     const c10::optional<Tensor>& bias,
//     Scalar exponent);

// Tensor linear_leaky_relu(
//     const Tensor& input,
//     const Tensor& weight,
//     const c10::optional<Tensor>& bias,
//     Scalar negative_slope);

// Tensor linear_hardtanh(
//     const Tensor& input,
//     const Tensor& weight,
//     const c10::optional<Tensor>& bias,
//     Scalar minval,
//     Scalar maxval);

// Tensor linear_elu(
//     const Tensor& input,
//     const Tensor& weight,
//     const c10::optional<Tensor>& bias,
//     Scalar alpha,
//     Scalar scale,
//     Scalar input_scale);

} // namespace AtenIpexTypeXPU
} // namespace at
