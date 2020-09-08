#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class Runtime final {
 public:
  enum class Type {
    Debug,
    Release,
  };

  explicit Runtime(Type type);
  Runtime(const Runtime&) = delete;
  Runtime(Runtime&&) = default;
  Runtime& operator=(const Runtime&) = delete;
  Runtime& operator=(Runtime&&) = default;
  ~Runtime() = default;

  inline VkInstance instance() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(instance_);
    return instance_.get();
  }

  typedef std::function<bool (const Adapter&)> Selector;
  Adapter select(const Selector& selector);

 private:
  class Debug final {
   public:
    explicit Debug(VkInstance instance);
    void operator()(VkDebugReportCallbackEXT debug_report_callback) const;

   private:
    VkInstance instance_;
  };

 private:
  // Construction and destruction order matters.  Do not move members around.
  Handle<VkInstance, decltype(&VK_DELETER(Instance))> instance_;
  Handle<VkDebugReportCallbackEXT, Debug> debug_report_callback_;
};

bool available();
Runtime* runtime();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
