#include <BlasImpl.h>
#include <detail/oneDNN.h>


namespace at {
namespace native::xpu {
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
    onednn::Attr attr = func();
    Tensor _bias = bias.has_value() ? bias.value() : at::Tensor();
    Tensor _input =
        input.dim() <= 2 ? input : input.contiguous();
    return impl::matmul_fusion_variants(
        result, _input, weight, false, attr, is_fused_, _bias);
  }

  Tensor& call(
      Tensor& result,
      const Tensor& input,
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      onednn::Attr attr) {
    Tensor _bias = bias.has_value() ? bias.value() : at::Tensor();
    Tensor _input =
        input.dim() <= 2 ? input : input.contiguous();
    return impl::matmul_fusion_variants(
        result, _input, weight, /*trans*/ true, attr, is_fused_, _bias);
  }

  bool is_fused() {
    return is_fused_;
  }

  bool is_fused_;
};


} // namespace native::xpu
} // namespace at
