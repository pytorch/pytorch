#pragma once

#ifdef USE_VULKAN_API

#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class VulkanPackedContext {
 protected:
  c10::impl::GenericList packed_;

 public:
  VulkanPackedContext() : packed_{c10::AnyType::get()} {}
  VulkanPackedContext(const VulkanPackedContext&) = default;
  VulkanPackedContext(VulkanPackedContext&&) = default;

  inline const c10::IValue get_val(int64_t i) const {
    return packed_.get(i);
  }

  inline void set_val(int64_t i, const c10::IValue& val) const {
    return packed_.set(i, val);
  }

  virtual const c10::impl::GenericList unpack() const = 0;

  virtual ~VulkanPackedContext() = default;
};

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
