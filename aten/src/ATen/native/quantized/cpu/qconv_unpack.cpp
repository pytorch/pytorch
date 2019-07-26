#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>

namespace at {
namespace native {
namespace {

/*
 * QConvPackWeightInt8 expects its input tensor to be in shape
 * [output_channels, kernel_height, kernel_width, input_channels/Groups]
 * Therefore, the unpacking of packed weight tensor using QConvUnpackWeightsInt8
 * results in a tensor of the same shape.
 */
class QConvUnpackWeightsInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  Tensor operator()(Tensor packed_weights) {
    // Pull out the packed weight instance from the owning tensor.
    auto& pack_ptr =
        cpp_custom_type_hack::cast<PackedConvWeight>(packed_weights);
    auto packed_weights_p = pack_ptr.w.get();

    // output channels
    int output_channels = packed_weights_p->outputChannels();
    int input_channels = packed_weights_p->inputChannels();
    int groups = packed_weights_p->groups();
    // R (kernel height)
    int kernel_h = pack_ptr.kernel[0];
    // S (kernel width)
    int kernel_w = pack_ptr.kernel[1];

    int C_per_G = input_channels / groups;

    // Tensor for unpacked weights
    // Unpacked format would be KRS(C/G)
    auto unpacked_weights = _empty_affine_quantized(
        {output_channels, kernel_h, kernel_w, C_per_G},
        device(kCPU).dtype(kQInt8),
        pack_ptr.w_scale,
        pack_ptr.w_zp);
    int8_t* unpacked_weights_p =
        reinterpret_cast<int8_t*>(unpacked_weights.data<c10::qint8>());

    packed_weights_p->unpack(unpacked_weights_p);

    return unpacked_weights;
  }
#else // USE_FBGEMM
  Tensor operator()(Tensor /* weight */
  ) {
    // We make a strong guarantee that models using these operators will have
    // the same numerics across different machines. Therefore, we do not provide
    // a fallback path and rather fail loudly if we cannot run FBGEMM.
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry = c10::RegisterOperators().op(
    "quantized::fbgemm_conv_unpack(Tensor packed_weights)"
    " -> Tensor unpacked_weights",
    c10::RegisterOperators::options().kernel<QConvUnpackWeightsInt8>(
        CPUTensorId()));

} // namespace
} // namespace native
} // namespace at
