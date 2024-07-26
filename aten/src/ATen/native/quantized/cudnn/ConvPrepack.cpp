#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include <array>
#include <vector>

template <int kSpatialDim = 2>
int register_conv_params();

extern template int register_conv_params<2>();
extern template int register_conv_params<3>();

template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightCudnn<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  // TODO: need to check out to implement groups for conv operator in Conv.cpp
  TORCH_CHECK(groups == 1, "Quantized cudnn conv2d is currently limited to groups = 1; received groups =", groups);
  TORCH_CHECK(weight.qscheme() == c10::kPerTensorAffine, "Unsupported qscheme: ", toString(weight.qscheme()));
  TORCH_CHECK(
      kSpatialDim == 2,  // 1D is packed as 2d, hence we don't need other checks
      "cuDNN packing only supports 2D convolution.");
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
  TORCH_CHECK(!transpose, "cudNN quantized conv prepack expects transpose = false")
  const int num_unpadded_output_channels = weight.size(0);
  const auto qtype = weight.qscheme();
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.value().size(0) == num_unpadded_output_channels,
        "bias should have K elements: " + std::to_string(num_unpadded_output_channels));
    // TODO: we create a broadcasted_bias tensor later so I think we don't need to make this contiguous here.
    // we will revisit this when nvidia adds proper support for broadcasting
    // bias_contig = bias->contiguous();
  }

  // cudnn v8.4.0 expects conv2d's int8 weight tensor's input and output channels to be a multiple of 4. if it is not
  // we need to explicitly pad it to a multiple of 4 ourselves as cudnn does not currently support padding.
  // TODO: when and if cudnn enables padding in their operators, we can remove padding on our end;
  // currently, limit padding support to groups=1 (ungrouped conv)
  // TODO: implement this for groups > 1
  auto num_input_channels = weight.size(1);
  int8_t num_output_slices2pad = (4 - num_unpadded_output_channels % 4) % 4;
  int8_t num_input_slices2pad = (4 - num_input_channels % 4) % 4;
  if (num_output_slices2pad != 0 || num_input_slices2pad != 0) {
    // the second argument is an initializer list of padded values. there are 2 values for each dimension.
    // refer to https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html for more details
    weight = at::pad(weight, {0, 0, 0, 0, 0, num_input_slices2pad, 0, num_output_slices2pad}, "constant", 0);
    if (bias.has_value()) {
      bias.value() = at::pad(bias.value(), {0, num_output_slices2pad}, "constant", 0);
    }
  }

  auto ret_ptr = c10::make_intrusive<PackedConvWeightCudnn<kSpatialDim>>(
          weight.to(c10::MemoryFormat::ChannelsLast), // TODO: this assumes 2D I think. make it more general?
          bias,
          stride,
          padding,
          output_padding,
          dilation,
          groups,
          transpose,
          qtype,
          num_unpadded_output_channels);
  return ret_ptr;
}

template
c10::intrusive_ptr<ConvPackedParamsBase<2>> PackedConvWeightCudnn<
    2>::
    prepack(
        at::Tensor weight,
        std::optional<at::Tensor> bias_in,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose);

namespace at {
namespace native {
namespace {

template <int kSpatialDim = 2>
class QConvPackWeightInt8Cudnn final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_conv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    torch::List<int64_t> output_padding;
    output_padding.reserve(kSpatialDim);
    for (C10_UNUSED const auto idx : c10::irange(kSpatialDim)) {
      output_padding.push_back((int64_t)0);
    }
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> _run(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    return PackedConvWeightCudnn<kSpatialDim>::prepack(
        weight, bias, stride, padding, output_padding, dilation, groups,
        transpose);
  }
};

class QConv1dPackWeightInt8Cudnn final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_conv(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    const torch::List<int64_t> output_padding({0});
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> _run(
      Tensor weight,
      std::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    if (weight.dim() == 3) {
      // we currently use conv2d kernel for conv1d by making the input and weight tensors
      // 4D rather than 3D. we add a dummy width dimension of size 1
      // out channels, in channels / groups, L -> out channels, in channels / groups, 1, L
      weight = weight.unsqueeze(-2);
    }
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    output_padding = quant_utils::MakeArgForConv1d(output_padding, 0);
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);

    return PackedConvWeightCudnn<2>::prepack(
        weight, bias, stride, padding, output_padding, dilation, groups,
        transpose);
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  register_conv_params<2>();
  register_conv_params<3>();
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_prepack"), TORCH_FN(QConv1dPackWeightInt8Cudnn::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_prepack"), TORCH_FN(QConvPackWeightInt8Cudnn<2>::run_conv));
}

} // namespace
} // namespace native
} // namespace at

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA
