#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum Conv2dMethod {
  Conv2dDepthwise,
  Conv2dPointwise,
  Conv2dSlidingWindow,
  Conv2dWinograd_2_3,
};

//  private:
//   packed
//     vTensor v_weight
//     vTensor v_bias
//     std::array<int64_t, 4> filter
//     std::array<int64_t, 2> stride
//     std::array<int64_t, 2> padding
//     std::array<int64_t, 2> dilation
//     int32_t groups
//     float output_min
//     float output_max

//   unpacked
//     Tensor weight
//     c10::optional<Tensor> bias
//     std::vector<int64_t> filter
//     std::vector<int64_t> stride
//     std::vector<int64_t> padding
//     std::vector<int64_t> dilation
//     int64_t groups
//     c10::optional<Scalar> output_min
//     c10::optional<Scalar> output_max

VulkanOpContext conv2d_context_create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const c10::optional<Scalar>& output_min = c10::nullopt,
    const c10::optional<Scalar>& output_max = c10::nullopt);

Tensor conv2d_context_run(
    const Tensor& input_arg,
    const c10::impl::GenericList& packed_context,
    const c10::impl::GenericList& unpacked_context);

Tensor run_conv2d_clamp_context(
    const Tensor& input,
    const c10::intrusive_ptr<VulkanOpContext>& context);

c10::intrusive_ptr<VulkanOpContext> create_conv2d_clamp_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max);

// Backwards compatibility
class Conv2dOpContext final : public torch::jit::CustomClassHolder {
 public:
  static Conv2dOpContext create(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      bool transposed,
      IntArrayRef output_padding,
      int64_t groups,
      const c10::optional<Scalar>& output_min = c10::nullopt,
      const c10::optional<Scalar>& output_max = c10::nullopt);

  using State = std::tuple<
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      c10::optional<Scalar>,
      c10::optional<Scalar>>;

  Tensor run(const Tensor& input) const;
  State unpack() const;

 private:
  explicit Conv2dOpContext(VulkanOpContext vulkan_context);
  VulkanOpContext vulkan_context_;
};

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& context);

c10::intrusive_ptr<Conv2dOpContext> conv2d_clamp_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
