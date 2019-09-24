#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qmkldnn_utils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/quantized/Quantizer.h>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedConvWeight);
#if AT_MKLDNN_ENABLED()
CAFFE_KNOWN_TYPE(PackedConvWeightQmkldnn);
#endif
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedConvWeightsQnnp);
#endif // USE_PYTORCH_QNNPACK
} // namespace caffe2

namespace at {
namespace native {
namespace {
class QConvPackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_PYTORCH_QNNPACK
  at::Tensor qnnpack_conv_prepack(
      Tensor weight,
      c10::optional<Tensor> bias_in,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    TORCH_CHECK(
        weight.ndimension() == 4,
        "quantized::conv_prepack (qnnpack): Weights are expected to have 4 dimensions");
    const auto qtype = weight.qscheme();
    TORCH_CHECK(
        weight.qscheme() == kPerTensorAffine,
        "quantized::conv_prepack (qnnpack): only supports Per Tensor Quantization Scheme")
    TORCH_CHECK(
        stride.size() == 2,
        "quantized::conv_prepack (qnnpack): 2D convolution only");
    TORCH_CHECK(
        padding.size() == 2,
        "quantized::conv_prepack (qnnpack): Specify top/left padding only. \
       bottom/right padding assumed to be equal to top/left");
    TORCH_CHECK(
        dilation.size() == 2,
        " quantized::conv_prepack (qnnpack): 2D convolution only");

    initQNNPACK();

    // QNNPACK expects weights to be of the format {out_c, kH, kW, in_c/groups},
    // but PyTorch lays them out as {out_c, in_c/groups, kH, kW}
    const size_t out_ch = weight.size(0);
    const size_t in_ch = weight.size(1) * groups;
    const uint32_t kernel_h = weight.size(2);
    const uint32_t kernel_w = weight.size(3);

    Tensor bias_fp32;
    if (bias_in.has_value()) {
      bias_fp32 = bias_in.value();
    } else {
      bias_fp32 = at::zeros(out_ch, weight.options().dtype(at::kFloat));
    }
    TORCH_CHECK(
        !bias_fp32.defined() || (bias_fp32.ndimension() == 1 && bias_fp32.size(0) == out_ch),
        "quantized::conv_prepack (qnnpack): expected bias to be 1-dimensional with ",
        out_ch,
        " elements",
        ", but got bias of size ",
        bias_fp32.sizes(),
        " instead");

    uint32_t stride_h = stride[0];
    uint32_t stride_w = stride[1];
    uint32_t pad_t = padding[0];
    uint32_t pad_l = padding[1];
    uint32_t dilation_h = dilation[0];
    uint32_t dilation_w = dilation[1];

    qnnpack::conv_param_t conv_p(
        {kernel_w, kernel_h},
        {stride_w, stride_h},
        {dilation_w, dilation_h},
        {pad_t, pad_l, pad_t, pad_l},
        groups,
        in_ch,
        out_ch,
        weight.q_zero_point(),
        weight.q_scale(),
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max());

    auto weight_contig = weight.contiguous(MemoryFormat::ChannelsLast);
    auto weight_zp = weight.q_zero_point() + 128;

    int8_t* w_data = (int8_t*)weight_contig.data_ptr<c10::qint8>();
    Tensor qnnp_weight = at::_empty_affine_quantized(
        weight_contig.sizes(),
        at::device(kCPU).dtype(kQUInt8),
        weight.q_scale(),
        weight_zp);
    auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
    auto wt_numel = weight_contig.numel();
    for (int i = 0; i < wt_numel; ++i) {
      qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
    }
    // We set the pre-packed conv weights to nullptr below as we call pre-pack
    // during the first invocation of operator run. Refer to qconv.cpp for more
    // details. TODO Update to actually call pre-pack here once bias is removed
    // from pre-packing step.
    auto wt_ptr = guts::make_unique<PackedConvWeightsQnnp>(
        PackedConvWeightsQnnp{nullptr, /* PrePackConvWeights */
                              weight_contig, /* int8_t weight */
                              bias_fp32.contiguous(), /* fp32 bias */
                              c10::nullopt, /* input_scale */
                              {kernel_h, kernel_w},
                              weight.q_scale(),
                              weight_zp});

    return cpp_custom_type_hack::create(std::move(wt_ptr), weight.options());
  }
#endif // USE_PYTORCH_QNNPACK
  Tensor operator()(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
#if AT_MKLDNN_ENABLED()
    bool is_zero = weight.qscheme() == c10::kPerChannelAffine
        ? is_zeros<>(weight.q_per_channel_zero_points())
        : weight.q_zero_point() == 0;

    // TODO: Will remove it after mkldnn fix the bug in 1x1 conv op.now mkldnn
    // will output wrong result when groups > 1,and IC,OC != 8x.
    bool is_1x1_conv =
        (groups > 1 && weight.size(2) == 1 && weight.size(3) == 1);

    if ((ctx.qEngine() == at::kQMKLDNN) && is_zero && !is_1x1_conv &&
        (weight.scalar_type() == at::kQInt8)) {
      return mkldnn_conv_prepack(weight, bias, stride, padding, dilation, groups);
    } else if (ctx.qEngine() == at::kQMKLDNN) {
      return fbgemm_conv_prepack(weight, bias, stride, padding, dilation, groups);
    }
#endif
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_conv_prepack(weight, bias, stride, padding, dilation, groups);
    }

#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return qnnpack_conv_prepack(
          weight, bias, stride, padding, dilation, groups);
    }
#endif
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv_prepack ",
        toString(ctx.qEngine()));
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::conv_prepack",
    c10::RegisterOperators::options().kernel<QConvPackWeightInt8>(
        TensorTypeId::QuantizedCPUTensorId));

} // namespace
} // namespace native
} // namespace at
