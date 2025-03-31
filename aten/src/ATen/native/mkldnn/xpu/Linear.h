#include <BlasImpl.h>
#include <detail/oneDNN.h>

namespace at::native::xpu {

struct LinearConverter {
  LinearConverter() {
    is_fused_ = false;
  }

  Tensor& call(
      Tensor& result,
      const Tensor& input,
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      onednn::Attr attr) {
    Tensor _bias = bias.has_value() ? bias.value() : at::Tensor();
    Tensor _input = input.dim() <= 2 ? input : input.contiguous();
    return impl::matmul_fusion_variants(
        result, _input, weight, /*trans*/ false, attr, is_fused_, _bias);
  }

  bool is_fused() {
    return is_fused_;
  }

  bool is_fused_;
};

} // namespace at::native::xpu
