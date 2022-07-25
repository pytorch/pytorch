#pragma once

#ifdef USE_VULKAN_API

#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class VulkanOpContext final : public torch::jit::CustomClassHolder {
 public:
  static VulkanOpContext create(c10::impl::GenericList source_args) {
    return VulkanOpContext(source_args);
  }

  explicit VulkanOpContext(c10::impl::GenericList source_args)
      : source_args_{source_args}, packed_args_{c10::AnyType::get()} {}

 private:
  c10::impl::GenericList source_args_;
  c10::impl::GenericList packed_args_;

 public:
  using State = c10::impl::GenericList;

  State get_state() const {
    return source_args_;
  }

  const c10::impl::GenericList& get_source_args() const {
    return source_args_;
  }

  const c10::impl::GenericList& get_packed_args() const {
    return packed_args_;
  }
};

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
