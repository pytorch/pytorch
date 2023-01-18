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

  inline const c10::IValue get_val(int64_t i) const {
    return packed_.get(i);
  }

  virtual const c10::impl::GenericList unpack() const = 0;

  virtual ~VulkanPackedContext() = default;
};

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
