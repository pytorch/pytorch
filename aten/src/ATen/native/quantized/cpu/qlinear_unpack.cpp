#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qmkldnn_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

namespace at {
namespace native {
namespace {

class QLinearUnpackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> qnnpack_linear_unpack(
      at::Tensor packed_weight) {
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedLinearWeightsQnnp>(packed_weight);
    return std::tuple<at::Tensor, c10::optional<Tensor>>(
        pack_ptr.orig_weight, pack_ptr.bias);
  }
#endif // USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> operator()(
      at::Tensor packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
#if AT_MKLDNN_ENABLED()
    if (cpp_custom_type_hack::is_type<PackedLinearWeightQmkldnn>(
            packed_weight)) {
      return mkldnn_linear_unpack(packed_weight);
    }
#endif
    if (cpp_custom_type_hack::is_type<PackedLinearWeight>(
            packed_weight)) {
      return fbgemm_linear_unpack(packed_weight);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear_unpack(packed_weight);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_unpack ",
        toString(ctx.qEngine()));
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::linear_unpack(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)",
    c10::RegisterOperators::options().kernel<QLinearUnpackWeightInt8>(
        TensorTypeId::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
