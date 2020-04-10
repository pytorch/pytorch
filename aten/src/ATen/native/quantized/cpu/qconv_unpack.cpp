#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <torch/custom_class.h>

template <int kSpatialDim>
torch::jit::class_<ConvPackedParamsBase<kSpatialDim>> register_conv_params();

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
class QConvUnpackWeightsInt8 final : public c10::OperatorKernel {
 public:
  std::tuple<at::Tensor, c10::optional<at::Tensor>> operator()(
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

static auto registry =
    c10::RegisterOperators()
        .op(c10::RegisterOperators::options()
            .schema("quantized::conv_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight) -> (Tensor W_origin, Tensor? B_origin)")
            .catchAllKernel<QConvUnpackWeightsInt8<2>>()) // conv_unpack is deprecated, please
        // use conv2d_unpack for 2D conv.
        .op(c10::RegisterOperators::options()
            .schema("quantized::conv2d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight) -> (Tensor W_origin, Tensor? B_origin)")
            .catchAllKernel<QConvUnpackWeightsInt8<2>>()) // We use  conv2d_unpack to be
                                           // consistent with conv3d_unpack
        .op(c10::RegisterOperators::options()
            .schema("quantized::conv3d_unpack(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight) -> (Tensor W_origin, Tensor? B_origin)")
            .catchAllKernel<QConvUnpackWeightsInt8<3>>());

} // namespace
} // namespace native
} // namespace at
