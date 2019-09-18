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
      c10::optional<Tensor> bias,
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
         static_cast<int>(padding[1])},
        {static_cast<int>(dilation[0]), static_cast<int>(dilation[1])});

    auto weight_contig = weight.contiguous();
    const auto qtype = weight.qscheme();
    std::vector<int32_t> zero_points(1, 0);
    if (qtype == kPerTensorAffine) {
      zero_points[0] = weight.q_zero_point();
    } else if (qtype == kPerChannelAffine) {
      zero_points.resize(output_channels, 0);
      for (int i = 0; i < output_channels; ++i) {
        zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
      }
    }

    const int8_t* weight_ptr_int8 =
        reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());

    std::vector<int32_t> col_offsets(output_channels);
    // compute column offsets (Similar to
    // fbgemm::col_offsets_with_zero_pt_s8acc32_ref) please note that offsets
    // include the sum of columns as well as the scalar term weight_zero_point *
    // KDim
    int NDim = output_channels / groups;
    int KDim_per_group = kernel_h * kernel_w * input_channels_per_group;
    for (int g = 0; g < groups; ++g) {
      for (int j = 0; j < NDim; ++j) {
        int32_t sum = 0;
        for (int k = 0; k < KDim_per_group; ++k) {
          sum += weight_ptr_int8[(g * NDim + j) * KDim_per_group + k];
        }
        if (qtype == kPerTensorAffine) {
          col_offsets[g * NDim + j] = sum - zero_points[0] * KDim_per_group;
        } else {
          col_offsets[g * NDim + j] =
              sum - zero_points[g * NDim + j] * KDim_per_group;
        }
      }
    }

    std::vector<float> scales(1, 0.0);
    if (qtype == kPerTensorAffine) {
      scales[0] = weight.q_scale();
    } else if (qtype == kPerChannelAffine) {
      scales.resize(output_channels, 0.0);
      for (int i = 0; i < output_channels; ++i) {
        scales[i] = weight.q_per_channel_scales()[i].item<float>();
      }
    }
    c10::optional<at::Tensor> bias_contig;
    if (bias.has_value()) {
      Tensor bias_vec = bias.value();
      TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
      TORCH_CHECK(
          bias_vec.size(0) == output_channels,
          "bias should have K elements: " + std::to_string(output_channels));
      bias_contig = bias->contiguous();
    }
    auto ret_ptr = guts::make_unique<PackedConvWeight>(
        PackedConvWeight{guts::make_unique<fbgemm::PackWeightsForConv<2>>(
                             conv_p, weight_ptr_int8),
                         bias_contig,
                         col_offsets,
                         {kernel_h, kernel_w},
                         scales,
                         zero_points,
                         qtype});
    // TODO: we will need to replace this with torchscript classes at a later
    // point.
    return cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
  }
#else // USE_FBGEMM
  Tensor operator()(
      Tensor, /* weight */
      c10::optional<Tensor>, /* bias */
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
    "quantized::conv_prepack",
    c10::RegisterOperators::options().kernel<QConvPackWeightInt8>(
        TensorTypeId::QuantizedCPUTensorId));

} // namespace
} // namespace native
} // namespace at
