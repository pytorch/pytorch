#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/library.h>

int register_linear_params();

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightCudnn::prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias) {
  TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine, "Unsupported qscheme: ", toString(weight.qscheme()));
  const auto output_channels = weight.size(0);
  const auto qtype = weight.qscheme();
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.value().size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
  }

  auto ret_ptr = c10::make_intrusive<PackedLinearWeightCudnn>(
          std::move(weight),
          std::move(bias),
          qtype);
  return ret_ptr;
}


namespace at::native {
namespace {

class QLinearPackWeightInt8Cudnn final {
 public:
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      at::Tensor weight,
      std::optional<Tensor> bias) {
      return PackedLinearWeightCudnn::prepack(std::move(weight), std::move(bias));
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  register_linear_params();
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack"), TORCH_FN(QLinearPackWeightInt8Cudnn::run));
}


} // namespace
} // namespace at::native


#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
