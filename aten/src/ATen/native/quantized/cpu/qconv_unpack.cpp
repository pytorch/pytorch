#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

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
      Tensor packed_weights) {
    auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
    if (ctx.qEngine() == at::QEngine::FBGEMM) {
      return fbgemm_conv_unpack(packed_weights);
    }
#endif

#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          kSpatialDim == 2,
          "quantized::conv2d_unpack (qnnpack): QNNPACK only supports Conv2d "
          "now.");
      return qnnpack_conv_unpack(packed_weights);
    }
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_unpack ",
        toString(ctx.qEngine()));
  }

 private:
#ifdef USE_FBGEMM
  std::tuple<Tensor, c10::optional<Tensor>> fbgemm_conv_unpack(
      Tensor packed_weights) {
    // Pull out the packed weight instance from the owning tensor.
    auto& pack_ptr = cpp_custom_type_hack::cast<PackedConvWeight<kSpatialDim>>(
        packed_weights);
    auto* packed_weights_p = pack_ptr.w.get();

    // output channels
    const int output_channels = packed_weights_p->outputChannels();
    const int input_channels = packed_weights_p->inputChannels();
    const int groups = packed_weights_p->groups();

    const int kernel_d = kSpatialDim == 2 ? 1 : pack_ptr.kernel[0];
    // R (kernel height)
    const int kernel_h = pack_ptr.kernel[kSpatialDim - 2];
    // S (kernel width)
    const int kernel_w = pack_ptr.kernel[kSpatialDim - 1];

    const int C_per_G = input_channels / groups;

    // Tensor for unpacked weights
    // Unpacked format would be physical KRS(C/G) but logical KCRS (channels
    // first) because that's how
    // ChannelsLast3d is not available now.FBGEMM stores the weights
    // TODO: Unify 2d and 3d when ChannelsLast3d is ready.
    Tensor unpacked_weights;
    if (pack_ptr.q_scheme == kPerTensorAffine) {
      unpacked_weights = kSpatialDim == 2
          ? _empty_affine_quantized(
                {output_channels, C_per_G, kernel_h, kernel_w},
                device(kCPU).dtype(kQInt8).memory_format(MemoryFormat::ChannelsLast),
                pack_ptr.w_scale[0],
                pack_ptr.w_zp[0])
          : fbgemm_utils::MakeEmptyAffineQuantizedChannelsLast3dTensor(
                output_channels,
                C_per_G,
                kernel_d,
                kernel_h,
                kernel_w,
                device(kCPU).dtype(kQInt8),
                pack_ptr.w_scale[0],
                pack_ptr.w_zp[0]);
    } else if (pack_ptr.q_scheme == kPerChannelAffine) {
      auto scales = from_blob(
          pack_ptr.w_scale.data(),
          pack_ptr.w_scale.size(),
          device(kCPU).dtype(kFloat));
      auto zero_points = from_blob(
          pack_ptr.w_zp.data(), pack_ptr.w_zp.size(), device(kCPU).dtype(kInt));
      unpacked_weights = kSpatialDim == 2
          ? _empty_per_channel_affine_quantized(
                {output_channels, C_per_G, kernel_h, kernel_w},
                scales.toType(kDouble),
                zero_points.toType(kLong),
                0, /* The output channel axis is 0 */
                device(kCPU).dtype(kQInt8),
                MemoryFormat::ChannelsLast)
          : fbgemm_utils::
                MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
                    output_channels,
                    C_per_G,
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    device(kCPU).dtype(kQInt8),
                    scales.toType(kDouble),
                    zero_points.toType(kLong));
    } else {
      TORCH_CHECK(false, "Unsupported qscheme: ", toString(pack_ptr.q_scheme));
    }
    int8_t* unpacked_weights_p =
        reinterpret_cast<int8_t*>(unpacked_weights.data_ptr<c10::qint8>());
    packed_weights_p->unpack(unpacked_weights_p);

    return std::tuple<Tensor, c10::optional<Tensor>>(
        unpacked_weights, pack_ptr.bias);
  }
#endif

#ifdef USE_PYTORCH_QNNPACK
  std::tuple<at::Tensor, c10::optional<Tensor>> qnnpack_conv_unpack(
      at::Tensor packed_weight) {
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedConvWeightsQnnp>(packed_weight);
    return std::tuple<at::Tensor, c10::optional<Tensor>>(
        pack_ptr.orig_weight, pack_ptr.bias);
  }
#endif
};

static auto registry =
    c10::RegisterOperators()
        .op("quantized::conv_unpack(Tensor packed_weights)"
            " -> (Tensor unpacked_weights, Tensor? B_origin)",
            c10::RegisterOperators::options().kernel<QConvUnpackWeightsInt8<2>>(
                DispatchKey::CPUTensorId)) // conv_unpack is deprecated, please
                                            // use conv2d_unpack for 2D conv.
        .op("quantized::conv2d_unpack(Tensor packed_weights)"
            " -> (Tensor unpacked_weights, Tensor? B_origin)",
            c10::RegisterOperators::options().kernel<QConvUnpackWeightsInt8<2>>(
                DispatchKey::CPUTensorId)) // We use  conv2d_unpack to be
                                            // consistent with conv3d_unpack
        .op("quantized::conv3d_unpack",
            c10::RegisterOperators::options().kernel<QConvUnpackWeightsInt8<3>>(
                DispatchKey::CPUTensorId));

} // namespace
} // namespace native
} // namespace at
