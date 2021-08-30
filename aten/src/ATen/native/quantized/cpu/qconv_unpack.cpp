#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>

#ifdef USE_FBGEMM
template <int kSpatialDim>



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
      
      weight = weight.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
      return std::tuple<at::Tensor, c10::optional<at::Tensor>>(weight, bias);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      
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


TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  // conv_unpack is deprecated, please use conv2d_unpack for 2D conv.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
  // We use  conv2d_unpack to be consistent with conv3d_unpack
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_unpack"), TORCH_FN(QConv1dUnpackWeightsInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_unpack"), TORCH_FN(QConvUnpackWeightsInt8<2>::run));
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
