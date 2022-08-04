#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum Conv2dMethod {
  Conv2dDepthwise,
  Conv2dPointwise,
  Conv2dSlidingWindow,
  TConv2dSlidingWindow,
  QConv2dDepthwise,
  QConv2dPointwise,
  QConv2dSlidingWindow,
};

class Conv2dPackedContext final : virtual public VulkanPackedContext,
                                  public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;

 public:
  Conv2dPackedContext(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      const IntArrayRef stride_arg,
      const IntArrayRef padding_arg,
      const IntArrayRef dilation_arg,
      const bool transposed,
      const bool quantized,
      const IntArrayRef output_padding_arg,
      const int64_t groups,
      const c10::optional<Scalar>& output_min = c10::nullopt,
      const c10::optional<Scalar>& output_max = c10::nullopt);

  static Conv2dPackedContext pack(c10::impl::GenericList);

  const c10::impl::GenericList unpack() const override {
    return unpacked_;
  }
};

c10::intrusive_ptr<Conv2dPackedContext> create_conv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min = c10::nullopt,
    const c10::optional<Scalar>& output_max = c10::nullopt);

Tensor run_conv2d_context(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& context);

c10::intrusive_ptr<Conv2dPackedContext> create_tconv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min = c10::nullopt,
    const c10::optional<Scalar>& output_max = c10::nullopt);

Tensor run_tconv2d_context(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& context);

c10::intrusive_ptr<Conv2dPackedContext> create_qconv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min = c10::nullopt,
    const c10::optional<Scalar>& output_max = c10::nullopt);

Tensor run_qconv2d_context(
    const Tensor& input_arg,
    double scale,
    int64_t zero_point,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context);

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
  explicit Conv2dOpContext(Conv2dPackedContext conv_context);
  Conv2dPackedContext conv_context_;
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
