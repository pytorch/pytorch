#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qmkldnn_utils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/quantized/Quantizer.h>
#include <algorithm>
#include <vector>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedLinearWeight);
#if AT_MKLDNN_ENABLED()
CAFFE_KNOWN_TYPE(PackedLinearWeightQmkldnn);
#endif
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedLinearWeightsQnnp);
#endif // USE_PYTORCH_QNNPACK
} // namespace caffe2

namespace at {
namespace native {
namespace {

class QLinearPackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  at::Tensor qnnpack_linear_prepack(
      at::Tensor weight,
      c10::optional<Tensor> bias_in) {
    TORCH_CHECK(
        weight.dim() == 2,
        "quantized::linear_prepack (qnnpack): Weight tensor rank should be == 2");
    TORCH_CHECK(
        weight.qscheme() == kPerTensorAffine,
        "quantized::linear_prepack (qnnpack) only supports Per Tensor Quantization Scheme")

    int64_t rows_w = weight.size(0);
    Tensor bias_fp32;
    if (bias_in.has_value()) {
      bias_fp32 = bias_in.value();
    } else {
      bias_fp32 = at::zeros(rows_w, weight.options().dtype(at::kFloat));
    }
    TORCH_CHECK(
        !bias_fp32.defined() || (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == rows_w),
        "quantized::linear_prepack (qnnpack): Given weight of size ",
        weight.sizes(),
        ", expected bias to be 1-dimensional with ",
        rows_w,
        " elements",
        ", but got bias of size ",
        bias_fp32.sizes(),
        " instead");

    Tensor weight_contig = weight.contiguous();
    auto weight_zp = weight.q_zero_point() + 128;

    int8_t* inp_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
    Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(kCPU).dtype(kQUInt8),
        weight.q_scale(),
        weight_zp);
    auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
    auto wt_numel = weight_contig.numel();
    for (int i = 0; i < wt_numel; ++i) {
      qnnp_w_data[i] = static_cast<c10::quint8>(inp_data[i] + 128);
    }
    initQNNPACK();

    // We set the pre-packed linear weights to nullptr below as we call pre-pack
    // during the first invocation of operator run. Refer to qlinear.cpp for more
    // details. TODO Update to actually call pre-pack here once bias is removed
    // from pre-packing step.
    auto wt_ptr = guts::make_unique<PackedLinearWeightsQnnp>(
        PackedLinearWeightsQnnp{nullptr,
                                weight_contig, /* int8_t weight */
                                bias_fp32.contiguous(), /* fp32 bias */
                                c10::nullopt, /* input_scale */
                                weight.q_scale(),
                                weight_zp});
    return cpp_custom_type_hack::create(std::move(wt_ptr), weight.options());
  }
#endif
  at::Tensor operator()(at::Tensor weight, c10::optional<Tensor> bias) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
#if AT_MKLDNN_ENABLED()
    bool is_zero = weight.qscheme() == c10::kPerChannelAffine
        ? is_zeros<>(weight.q_per_channel_zero_points())
        : weight.q_zero_point() == 0;

    if ((ctx.qEngine() == at::kQMKLDNN) &&
          is_zero && (weight.scalar_type() == at::kQInt8)) {
      return mkldnn_linear_prepack(weight, bias);
    } else if(ctx.qEngine() == at::kQMKLDNN) {
      return fbgemm_linear_prepack(weight, bias);
    }
#endif
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_linear_prepack(weight, bias);
    }
#endif
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_linear_prepack(weight, bias);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack ",
        toString(ctx.qEngine()));
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::linear_prepack(Tensor W, Tensor? B=None) -> Tensor W_prepack",
    c10::RegisterOperators::options().kernel<QLinearPackWeightInt8>(
        TensorTypeId::QuantizedCPUTensorId));
} // namespace
} // namespace native
} // namespace at
