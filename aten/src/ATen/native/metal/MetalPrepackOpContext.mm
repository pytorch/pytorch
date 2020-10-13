#import <ATen/native/metal/MetalConvolution.h>
#import <ATen/native/metal/MetalPrepackOpContext.h>
#import <ATen/native/metal/MetalUtils.h>
#import <ATen/native/metal/mpscnn/MPSCNNOps.h>

#include <torch/script.h>

namespace at {
namespace native {
namespace metal {

c10::intrusive_ptr<Conv2dOpContext> conv2d_prepack(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    c10::optional<Scalar> output_min,
    c10::optional<Scalar> output_max) {
  TORCH_CHECK(weight.dim() == 4);
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(weight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

c10::intrusive_ptr<Conv2dOpContext> unpack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    c10::optional<Scalar> output_min,
    c10::optional<Scalar> output_max) {
  const Tensor weightContig = weight.contiguous();
  const auto ws = weightContig.sizes();
  auto packed_buffer = permuteWeights(weightContig.data_ptr<float>(), ws.vec());
  auto packedWeight = at::empty(ws);
  int64_t size_bytes = at::prod_intlist(ws) * sizeof(float);
  memcpy(packedWeight.data_ptr(), packed_buffer.data(), size_bytes);
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(packedWeight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

Tensor conv2d_prepack_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& op_context) {
  return conv2d_prepack_run_impl(*op_context, input);
}

Tensor copy_to_host(const Tensor& input) {
  return mpscnn::copy_to_host(input);
}

} // namespace metal
} // namespace native
} // namespace at
