#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/quantized/Quantizer.h>

namespace caffe2 {
#ifdef USE_FBGEMM
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedConvWeight);
#endif
} // namespace caffe2

namespace at {
namespace native {
namespace {
class QConvPackWeightInt8 final : public c10::OperatorKernel {
 public:
#ifdef USE_FBGEMM
  Tensor operator()(
      Tensor weight,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    TORCH_CHECK(
        weight.ndimension() == 4, "Weights are expected to have 4 dimensions");

    TORCH_CHECK(stride.size() == 2, "2D convolution only");
    TORCH_CHECK(
        padding.size() == 2,
        "Specify top/left padding only. \
        bottom/right padding assumed to be equal to top/left");
    TORCH_CHECK(dilation.size() == 2, "2D convolution only");
    TORCH_CHECK(
        (dilation[0] == 1 && dilation[1] == 1),
        "Currently dilation should be 1");
    // weights in KRS(C/G) format
    int output_channels = weight.size(0);
    int kernel_h = weight.size(1);
    int kernel_w = weight.size(2);
    int input_channels_per_group = weight.size(3);

    // mini-batch doesn't have any impact on how we pack weights
    // so we pass it as 1
    // Input image height/width also don't have any impact on how we pack
    // weights so we can pass any values
    fbgemm::conv_param_t<2> conv_p(
        1, // Mini-Batch
        input_channels_per_group * groups, // input channels
        output_channels,
        {28, 28}, // Image height and width
        groups,
        {kernel_h, kernel_w},
        {static_cast<int>(stride[0]), static_cast<int>(stride[1])},
        {static_cast<int>(padding[0]),
         static_cast<int>(padding[1]),
         static_cast<int>(padding[0]),
         static_cast<int>(padding[1])});

    // int NDim = output_channels / groups;
    // int KDim_per_group = kernel_h * kernel_w * input_channels_per_group;

    auto weight_contig = weight.contiguous();
    int32_t weight_zero_point_int32 = weight.q_zero_point();
    TORCH_CHECK(
        weight_zero_point_int32 == 0,
        "Only symmetric quantization is supported for weights yet");
    const int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_contig.data<c10::qint8>());

    std::vector<int32_t> col_offsets(output_channels);

    auto ret_ptr = guts::make_unique<PackedConvWeight>(
        PackedConvWeight{guts::make_unique<fbgemm::PackWeightsForConv<2>>(
                             conv_p, weight_ptr_int8),
                         col_offsets,
                         {kernel_h, kernel_w},
                         weight.q_scale(),
                         weight_zero_point_int32});
    // TODO: we will need to replace this with torchscript classes at a later
    // point.
    return cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
  }
#else // USE_FBGEMM
  Tensor operator()(
      Tensor, /* weight */
      torch::List<int64_t>, /* stride */
      torch::List<int64_t>, /* padding */
      torch::List<int64_t>, /* dilation */
      int64_t /* groups */
  ) {
    TORCH_CHECK(
        false, "This PyTorch installation was not built with FBGEMM operators");
  }
#endif // USE_FBGEMM
};

static auto registry = c10::RegisterOperators().op(
    "quantized::fbgemm_conv_prepack",
    c10::RegisterOperators::options().kernel<QConvPackWeightInt8>(
        QuantizedCPUTensorId()));

} // namespace
} // namespace native
} // namespace at
