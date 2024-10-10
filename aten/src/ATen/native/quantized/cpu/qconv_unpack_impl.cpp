#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <ATen/native/quantized/PackedParams.h>

#ifdef USE_FBGEMM
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<
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
              std::nullopt)
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
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(
      unpacked_weights, bias);
}

template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<
    2>::unpack();
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeight<
    3>::unpack();
#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<
    kSpatialDim>::unpack() {
  TORCH_CHECK(
      kSpatialDim == 2,
      "QNNPACK only supports conv2d_unpack right "
      "now.");
  TORCH_CHECK(
        orig_weight.defined(),
        "Cannot unpack weights. "
        "Call at::globalContext()::setReleaseOriginalWeights(false) before packing or loading to enable unpacking.");
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(orig_weight, bias);
}

template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<
    2>::unpack();
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsQnnp<
    3>::unpack();
#endif // USE_PYTORCH_QNNPACK

#if AT_ONEDNN_ENABLED()
template <int kSpatialDim>
std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<
    kSpatialDim>::unpack() {
  return std::tuple<at::Tensor, std::optional<at::Tensor>>(
      orig_weight_.clone(), orig_bias_);
}

template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<
    2>::unpack();
template std::tuple<at::Tensor, std::optional<at::Tensor>> PackedConvWeightsOnednn<
    3>::unpack();
#endif // #if AT_ONEDNN_ENABLED()
