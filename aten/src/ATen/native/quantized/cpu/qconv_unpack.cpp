#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <ATen/native/quantized/packed_params.h>

#ifdef USE_FBGEMM
template <int kSpatialDim>
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeight<
    kSpatialDim>::unpack() {
  auto* packed_weights_p = w.get();
  // output channels
  const int output_channels = packed_weights_p->outputChannels();
  const int input_channels = packed_weights_p->inputChannels();
  const int groups = packed_weights_p->groups();

  const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
  // R (kernel height)
  const int kernel_h = kernel[kSpatialDim - 2];
  // S (kernel width)
  const int kernel_w = kernel[kSpatialDim - 1];

  const int C_per_G = input_channels / groups;

  // Tensor for unpacked weights
  // Unpacked format would be physical KRS(C/G) but logical KCRS (channels
  // first) because that's how
  // ChannelsLast3d is not available now.FBGEMM stores the weights
  // TODO: Unify 2d and 3d when ChannelsLast3d is ready.
  at::Tensor unpacked_weights;
  if (q_scheme == c10::kPerTensorAffine) {
    unpacked_weights = kSpatialDim == 2
        ? at::_empty_affine_quantized(
              {output_channels, C_per_G, kernel_h, kernel_w},
              device(c10::kCPU)
                  .dtype(c10::kQInt8)
                  .memory_format(c10::MemoryFormat::ChannelsLast),
              w_scale[0],
              w_zp[0],
              c10::nullopt)
        : at::native::fbgemm_utils::
              MakeEmptyAffineQuantizedChannelsLast3dTensor(
                  output_channels,
                  C_per_G,
                  kernel_d,
                  kernel_h,
                  kernel_w,
                  device(c10::kCPU).dtype(c10::kQInt8),
                  w_scale[0],
                  w_zp[0]);
  } else if (q_scheme == c10::kPerChannelAffine) {
    TORCH_CHECK(
        !transpose(),
        "Per Channel Quantization is currently disabled for transposed conv");
    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kInt));
    unpacked_weights = kSpatialDim == 2
        ? at::_empty_per_channel_affine_quantized(
              {output_channels, C_per_G, kernel_h, kernel_w},
              scales.toType(c10::kDouble),
              zero_points.toType(c10::kLong),
              0, /* The output channel axis is 0 */
              device(c10::kCPU).dtype(c10::kQInt8),
              c10::MemoryFormat::ChannelsLast)
        : at::native::fbgemm_utils::
              MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
                  output_channels,
                  C_per_G,
                  kernel_d,
                  kernel_h,
                  kernel_w,
                  device(c10::kCPU).dtype(c10::kQInt8),
                  scales.toType(c10::kDouble),
                  zero_points.toType(c10::kLong));
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(q_scheme));
  }
  int8_t* unpacked_weights_p =
      reinterpret_cast<int8_t*>(unpacked_weights.data_ptr<c10::qint8>());
  packed_weights_p->unpack(unpacked_weights_p);
  if(transpose()){
    unpacked_weights =
        at::native::fbgemm_utils::TransposeConvTensorUnpackConversion<
            kSpatialDim>(unpacked_weights, groups);
  }
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      unpacked_weights, bias);
}

template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeight<
    2>::unpack();
template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeight<
    3>::unpack();
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
template <int kSpatialDim>
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightsQnnp<
    kSpatialDim>::unpack() {
  TORCH_CHECK(
      kSpatialDim == 2,
      "QNNPACK only supports conv2d_unpack right "
      "now.");
  TORCH_CHECK(
        orig_weight.defined(),
        "Cannot unpack weights. "
        "Call at::globalContext()::setReleaseOriginalWeights(false) before packing or loading to enable unpacking.");
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(orig_weight, bias);
}

template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightsQnnp<
    2>::unpack();
template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightsQnnp<
    3>::unpack();
#endif // USE_PYTORCH_QNNPACK

namespace at {
namespace native {
namespace {

/*
 * QConvPackWeightInt8 expects its input tensor to be in shape
 * [output_channels, kernel_height, kernel_width, input_channels/Groups]
 * Therefore, the unpacking of packed weight tensor using QConvUnpackWeightsInt8
 * results in a tensor of the same shape.
 */

template <int kSpatialDim = 2>
class QConvUnpackWeightsInt8 final {
 public:
  static std::tuple<at::Tensor, c10::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return packed_weight->unpack();
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          kSpatialDim == 2,
          "quantized::conv2d_unpack (qnnpack): QNNPACK only supports Conv2d "
          "now.");
      return packed_weight->unpack();
    }
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_unpack ",
        toString(ctx.qEngine()));
  }
};

class QConv1dUnpackWeightsInt8 final {
 public:
  static std::tuple<at::Tensor, c10::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight) {
    auto& ctx = at::globalContext();
    at::Tensor weight;
    c10::optional<at::Tensor> bias;
#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      std::tie(weight, bias) = packed_weight->unpack();
      weight = weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      return std::tuple<at::Tensor, c10::optional<at::Tensor>>(weight, bias);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      std::tie(weight, bias) = packed_weight->unpack();
      at::Tensor new_weight = weight.clone();
      new_weight = new_weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      return std::tuple<at::Tensor, c10::optional<at::Tensor>>(new_weight, bias);
    }
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv1d_unpack ",
        toString(ctx.qEngine()));
  }
};

template <int kSpatialDim = 2>
class QConvStride final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->stride();
  }
};

template <int kSpatialDim = 2>
class QConvPadding final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->padding();
  }
};

template <int kSpatialDim = 2>
class QConvOutputPadding final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->output_padding();
  }
};

template <int kSpatialDim = 2>
class QConvDilation final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->dilation();
  }
};

template <int kSpatialDim = 2>
class QConvGroups final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->groups();
  }
};

template <int kSpatialDim = 2>
class QConvTranspose final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->transpose();
  }
};

IValue
unpack_quantized_prepacked_sizes_conv2d(const IValue& ivalue) {
  auto params = ivalue.toCustomClass<ConvPackedParamsBase<2>>();
  at::Tensor weight;
  c10::optional<at::Tensor> bias;
  std::tie(weight, bias) = params->unpack();
  c10::optional<IntArrayRef> bias_sizes = c10::nullopt;
  if (bias && bias->defined()) {
    bias_sizes = bias->sizes();
  }
  return IValue(std::make_tuple(
      weight.sizes(),
      bias_sizes,
      params->stride(),
      params->padding(),
      params->dilation(),
      params->groups()));
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
  // We use  conv2d_unpack to be consistent with conv3d_unpack
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_unpack"), TORCH_FN(QConv1dUnpackWeightsInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_unpack_sizes"), TORCH_FN(unpack_quantized_prepacked_sizes_conv2d));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<3>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_stride"), TORCH_FN(QConvStride<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_padding"), TORCH_FN(QConvPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_output_padding"), TORCH_FN(QConvOutputPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_dilation"), TORCH_FN(QConvDilation<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_groups"), TORCH_FN(QConvGroups<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_transpose"), TORCH_FN(QConvTranspose<2>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_stride"), TORCH_FN(QConvStride<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_padding"), TORCH_FN(QConvPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_output_padding"), TORCH_FN(QConvOutputPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_dilation"), TORCH_FN(QConvDilation<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_groups"), TORCH_FN(QConvGroups<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_transpose"), TORCH_FN(QConvTranspose<3>::run));

  // ConvTranspose is the same, however, we want to have different name.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_unpack"), TORCH_FN(QConv1dUnpackWeightsInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<3>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_stride"), TORCH_FN(QConvStride<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_padding"), TORCH_FN(QConvPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_output_padding"), TORCH_FN(QConvOutputPadding<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_dilation"), TORCH_FN(QConvDilation<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_groups"), TORCH_FN(QConvGroups<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_transpose"), TORCH_FN(QConvTranspose<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_stride"), TORCH_FN(QConvStride<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_padding"), TORCH_FN(QConvPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_output_padding"), TORCH_FN(QConvOutputPadding<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_dilation"), TORCH_FN(QConvDilation<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_groups"), TORCH_FN(QConvGroups<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_transpose"), TORCH_FN(QConvTranspose<3>::run));
}

} // namespace
} // namespace native
} // namespace at
