#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include "c10/core/QScheme.h"
#include <array>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/native/quantized/packed_params.h>
#include <ATen/native/quantized/cudnn/cudnnpack_utils.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/library.h>

#include <c10/util/irange.h>

template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightCudnn<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        c10::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine, "Unsupported qscheme: ", toString(weight.qscheme()));
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ",
      kSpatialDim + 2,
      " dimensions");
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "stride should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "quantized::conv_prepack (cudnn): Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "quantized::conv_prepack (cudnn): dilation should contain ",
      kSpatialDim,
      " elements for ",
      kSpatialDim,
      "D convolution.");
  const int output_channels = transpose ? weight.size(1) * groups
                                        : weight.size(0);
  const auto qtype = weight.qscheme();

  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.value().size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    // we create a broadcasted_bias tensor later so I think we don't need to make this contiguous here.
    // we will revisit this when nvidia adds proper support for broadcasting
    // bias_contig = bias->contiguous();
  }

  return c10::make_intrusive<PackedConvWeightCudnn<kSpatialDim>>(
          weight,
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          groups,
          qtype);
}

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
