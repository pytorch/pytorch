#pragma once

#ifdef USE_VULKAN_API

#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class VulkanOpContext final : public torch::jit::CustomClassHolder {
 public:
  static VulkanOpContext create(
      c10::impl::GenericList packed_context,
      c10::impl::GenericList unpacked_context);
  using State = std::tuple<c10::impl::GenericList, c10::impl::GenericList>;
  State get_state() const;
  const c10::impl::GenericList& get_packed() const;
  const c10::impl::GenericList& get_unpacked() const;

 private:
  VulkanOpContext(
      c10::impl::GenericList packed_context,
      c10::impl::GenericList unpacked_context);
  c10::impl::GenericList packed_;
  c10::impl::GenericList unpacked_;
};

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
