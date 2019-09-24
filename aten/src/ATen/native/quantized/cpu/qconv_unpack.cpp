#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qmkldnn_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

namespace at {
namespace native {
namespace {

/*
 * QConvPackWeightInt8 expects its input tensor to be in shape
 * [output_channels, kernel_height, kernel_width, input_channels/Groups]
 * Therefore, the unpacking of packed weight tensor using QConvUnpackWeightsInt8
 * results in a tensor of the same shape.
 */
class QConvUnpackWeightsInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> qnnpack_conv_unpack(
      at::Tensor packed_weight) {
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedConvWeightsQnnp>(packed_weight);
    return std::tuple<at::Tensor, c10::optional<Tensor>>(
        pack_ptr.orig_weight, pack_ptr.bias);
  }
#endif
  std::tuple<at::Tensor, c10::optional<at::Tensor>> operator()(
      Tensor packed_weights) {
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
#if AT_MKLDNN_ENABLED()
    if (cpp_custom_type_hack::is_type<PackedConvWeightQmkldnn>(
            packed_weights)) {
      return mkldnn_conv_unpack(packed_weights);
    }
#endif
    if (cpp_custom_type_hack::is_type<PackedConvWeight>(
            packed_weights)) {
      return fbgemm_conv_unpack(packed_weights);
    }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_conv_unpack(packed_weights);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv_unpack ",
        toString(ctx.qEngine()));
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::conv_unpack(Tensor packed_weights)"
    " -> (Tensor unpacked_weights, Tensor? B_origin)",
    c10::RegisterOperators::options().kernel<QConvUnpackWeightsInt8>(
        TensorTypeId::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
