#include <ATen/ATen.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/mkldnn/quantization/Utils.h>
#include <torch/custom_class.h>
#include <torch/library.h>

torch::class_<LinearPackedParamsBase> register_linear_params();

#if AT_MKLDNN_ENABLED()
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedLinearWeightsMkldnn::unpack() {
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      orig_weight_, orig_bias_);
}
#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace {

class QLinearUnpackWeightInt8Mkldnn final {
 public:
  static std::tuple<at::Tensor, c10::optional<Tensor>> run(
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_unpack_mkldnn"), TORCH_FN(QLinearUnpackWeightInt8Mkldnn::run));
}

} // namespace
} // namespace native
} // namespace at
