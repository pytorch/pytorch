#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanOpContext.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

// packed
//   vTensor v_weight
//   vTensor v_bias

// unpacked
//   Tensor weight
//   c10::optional<Tensor> bias

VulkanOpContext linear_context_create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias);

Tensor linear_context_run(
    const Tensor& input_arg,
    const c10::impl::GenericList& packed_context,
    const c10::impl::GenericList& unpacked_context,
    const float alpha,
    const float beta,
    const std::string& op_name);

c10::intrusive_ptr<VulkanOpContext> create_linear_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias);

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<VulkanOpContext>& context);

// Backwards compatibility
class LinearOpContext final : public torch::jit::CustomClassHolder {
 public:
  static LinearOpContext create(
      const Tensor& weight,
      const c10::optional<Tensor>& bias);

  using State = std::tuple<Tensor, c10::optional<Tensor>>;

  Tensor run(
      const Tensor& input,
      float beta,
      float alpha,
      const std::string& op_name) const;
  State unpack() const;

 private:
  explicit LinearOpContext(VulkanOpContext vulkan_context);
  VulkanOpContext vulkan_context_;
};

c10::intrusive_ptr<LinearOpContext> linear_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias);

Tensor linear_run(
    const Tensor& input,
    const c10::intrusive_ptr<LinearOpContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
