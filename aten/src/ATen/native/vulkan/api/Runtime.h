#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// A Vulkan Runtime initializes a Vulkan instance and decouples the concept of
// Vulkan instance initialization from intialization of, and subsequent
// interactions with,  Vulkan [physical and logical] devices as a precursor to
// multi-GPU support.  The Vulkan Runtime can be queried for available Adapters
// (i.e. physical devices) in the system which in turn can be used for creation
// of a Vulkan Context (i.e. logical devices).  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
//

class Runtime final {
 public:
  enum class Type {
    Debug,
    Release,
  };

  explicit Runtime(Type type);
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = default;
  Runtime& operator=(Runtime&&) = default;
  ~Runtime() = default;

  VkInstance instance() const;

  typedef std::function<bool (const Adapter&)> Selector;
  Adapter select(const Selector& selector);

 private:
  class Debug final {
   public:
    explicit Debug(VkInstance);
    void operator()(VkDebugReportCallbackEXT) const;

   private:
    VkInstance instance_;
  };

 private:
  // Construction and destruction order matters.  Do not move members around.
  Handle<VkInstance, decltype(&VK_DELETER(Instance))> instance_;
  Handle<VkDebugReportCallbackEXT, Debug> debug_report_callback_;
};

Runtime* runtime();

//
// Impl
//

inline VkInstance Runtime::instance() const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(instance_);
  return instance_.get();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
