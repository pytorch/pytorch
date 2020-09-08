#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Vulkan Context holds onto all relevant Vulkan state as it pertains to our
// use of Vulkan in PyTorch.  The context is currently a global object, but
// technically it does not need to be if we were to make it explicit to the
// user.
//

class Context final {
 public:
  explicit Context(const Adapter& adapter);
  Context(const Context&) = delete;
  Context(Context&&) = default;
  Context& operator=(const Context&) = delete;
  Context& operator=(Context&&) = default;
  ~Context() = default;

  inline const Adapter& adapter() const {
    return adapter_;
  }

  inline VkDevice device() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_);
    return device_.get();
  }

  inline VkQueue queue() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(queue_);
    return queue_;
  }

  inline Command& command() {
    return command_;
  }

  inline Shader& shader() {
    return shader_;
  }

  inline Pipeline& pipeline() {
    return pipeline_;
  }

  inline Descriptor& descriptor() {
    return descriptor_;
  }

  inline Resource& resource() {
    return resource_;
  }

 private:
  // Construction and destruction order matters.  Do not move members around.
  Adapter adapter_;
  Handle<VkDevice, decltype(&VK_DELETER(Device))> device_;
  VkQueue queue_;
  Command command_;
  Shader shader_;
  Pipeline pipeline_;
  Descriptor descriptor_;
  Resource resource_;
};

bool available();
Context* context();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
