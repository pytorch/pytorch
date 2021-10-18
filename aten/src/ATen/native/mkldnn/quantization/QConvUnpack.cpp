#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/mkldnn/quantization/Utils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>

#if AT_MKLDNN_ENABLED()
template <int kSpatialDim>
std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightsMkldnn<
    kSpatialDim>::unpack() {
  std::cout << "quantized::conv_unpack_mkldnn (qengine = " << (int)at::globalContext().qEngine() << ")" << std::endl;
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      orig_weight_, orig_bias_);
}

template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightsMkldnn<
    2>::unpack();
template std::tuple<at::Tensor, c10::optional<at::Tensor>> PackedConvWeightsMkldnn<
    3>::unpack();
#endif // #if AT_MKLDNN_ENABLED()

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
class QConvUnpackWeightsInt8Mkldnn final {
 public:
  static std::tuple<at::Tensor, c10::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    auto& ctx = at::globalContext();

#if AT_MKLDNN_ENABLED()
    return packed_weight->unpack();
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_unpack ",
        toString(ctx.qEngine()));
  }
};

class QConv1dUnpackWeightsInt8Mkldnn final {
 public:
  static std::tuple<at::Tensor, c10::optional<at::Tensor>> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight) {
    auto& ctx = at::globalContext();
    at::Tensor weight;
    c10::optional<at::Tensor> bias;
#if AT_MKLDNN_ENABLED()
    std::tie(weight, bias) = packed_weight->unpack();
    at::Tensor new_weight = weight.clone();
    new_weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
    return std::tuple<at::Tensor, c10::optional<at::Tensor>>(new_weight, bias);
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv1d_unpack ",
        toString(ctx.qEngine()));
  }
};

template <int kSpatialDim = 2>
class QConvStrideMkldnn final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->stride();
  }
};

template <int kSpatialDim = 2>
class QConvPaddingMkldnn final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->padding();
  }
};

template <int kSpatialDim = 2>
class QConvOutputPaddingMkldnn final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->output_padding();
  }
};

template <int kSpatialDim = 2>
class QConvDilationMkldnn final {
 public:
  static torch::List<int64_t> run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->dilation();
  }
};

template <int kSpatialDim = 2>
class QConvGroupsMkldnn final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->groups();
  }
};

template <int kSpatialDim = 2>
class QConvTransposeMkldnn final {
 public:
  static int64_t run(
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight) {
    return packed_weight->transpose();
  }
};


TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_unpack_mkldnn"), TORCH_FN(QConvUnpackWeightsInt8Mkldnn<2>::run));
  // We use  conv2d_unpack to be consistent with conv3d_unpack
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_unpack_mkldnn"), TORCH_FN(QConv1dUnpackWeightsInt8Mkldnn::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_unpack_mkldnn"), TORCH_FN(QConvUnpackWeightsInt8Mkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_unpack_mkldnn"), TORCH_FN(QConvUnpackWeightsInt8Mkldnn<3>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_stride_mkldnn"), TORCH_FN(QConvStrideMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_padding_mkldnn"), TORCH_FN(QConvPaddingMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_output_padding_mkldnn"), TORCH_FN(QConvOutputPaddingMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_dilation_mkldnn"), TORCH_FN(QConvDilationMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_groups_mkldnn"), TORCH_FN(QConvGroupsMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_transpose_mkldnn"), TORCH_FN(QConvTransposeMkldnn<2>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_stride_mkldnn"), TORCH_FN(QConvStrideMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_padding_mkldnn"), TORCH_FN(QConvPaddingMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_output_padding_mkldnn"), TORCH_FN(QConvOutputPaddingMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_dilation_mkldnn"), TORCH_FN(QConvDilationMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_groups_mkldnn"), TORCH_FN(QConvGroupsMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_transpose_mkldnn"), TORCH_FN(QConvTransposeMkldnn<3>::run));

  // ConvTranspose is the same, however, we want to have different name.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_unpack_mkldnn"), TORCH_FN(QConv1dUnpackWeightsInt8Mkldnn::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_unpack_mkldnn"), TORCH_FN(QConvUnpackWeightsInt8Mkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_unpack_mkldnn"), TORCH_FN(QConvUnpackWeightsInt8Mkldnn<3>::run));

  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_stride_mkldnn"), TORCH_FN(QConvStrideMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_padding_mkldnn"), TORCH_FN(QConvPaddingMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_output_padding_mkldnn"), TORCH_FN(QConvOutputPaddingMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_dilation_mkldnn"), TORCH_FN(QConvDilationMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_groups_mkldnn"), TORCH_FN(QConvGroupsMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_transpose_mkldnn"), TORCH_FN(QConvTransposeMkldnn<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_stride_mkldnn"), TORCH_FN(QConvStrideMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_padding_mkldnn"), TORCH_FN(QConvPaddingMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_output_padding_mkldnn"), TORCH_FN(QConvOutputPaddingMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_dilation_mkldnn"), TORCH_FN(QConvDilationMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_groups_mkldnn"), TORCH_FN(QConvGroupsMkldnn<3>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_transpose_mkldnn"), TORCH_FN(QConvTransposeMkldnn<3>::run));
}

} // namespace
} // namespace native
} // namespace at
