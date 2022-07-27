#include <ATen/native/vulkan/ops/VulkanOpContext.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

VulkanOpContext::VulkanOpContext(
    c10::impl::GenericList packed_context,
    c10::impl::GenericList unpacked_context)
    : packed_(packed_context), unpacked_(unpacked_context) {}

VulkanOpContext VulkanOpContext::create(
    c10::impl::GenericList packed_context,
    c10::impl::GenericList unpacked_context) {
  return VulkanOpContext{packed_context, unpacked_context};
}

VulkanOpContext::State VulkanOpContext::get_state() const {
  return VulkanOpContext::State{packed_, unpacked_};
}

const c10::impl::GenericList& VulkanOpContext::get_packed() const {
  return packed_;
}

const c10::impl::GenericList& VulkanOpContext::get_unpacked() const {
  return unpacked_;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
