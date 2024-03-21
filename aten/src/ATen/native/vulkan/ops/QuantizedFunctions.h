#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Tensor quantize_per_tensor(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype);

Tensor quantize_per_tensor_tensor_qparams(
    const at::Tensor& input_arg,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    const c10::ScalarType dtype);

Tensor dequantize_helper(
    const at::Tensor& input_arg,
    const double scale,
    const int64_t zero_point,
    const c10::ScalarType dtype);

Tensor dequantize(const Tensor& self);

Tensor quantized_add(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point);

Tensor quantized_sub(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point);

Tensor quantized_mul(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point);

Tensor quantized_div(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point);

Tensor quantized_conv2d(
    const Tensor& input_,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    double out_scale,
    int64_t out_zero_point);

Tensor quantized_upsample_nearest2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const c10::optional<double> scales_h,
    const c10::optional<double> scales_w);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
