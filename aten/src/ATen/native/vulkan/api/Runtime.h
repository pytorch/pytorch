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

enum AdapterSelector {
  FIRST,
};

struct RuntimeConfiguration final {
  bool enableValidationMessages;
  bool initDefaultDevice;
  AdapterSelector defaultSelector;
};

class Runtime final {
 public:
  explicit Runtime(const RuntimeConfiguration config);

  // Do not allow copying. There should be only one global instance of this class.
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  Runtime(Runtime&&);
  Runtime& operator=(Runtime&&) = delete;

  ~Runtime();

  VkInstance instance() const;

  typedef std::function<size_t (const std::vector<Adapter>&)> Selector;
  size_t init_adapter(const Selector& selector);

  Adapter* get_adapter_p();
  Adapter& get_adapter();

  Adapter* get_adapter_p(size_t i);
  Adapter& get_adapter(size_t i);

  size_t default_adapter_i() const;

 private:
  VkInstance instance_;
  std::vector<Adapter> adapters_;
  size_t default_adapter_i_;

  VkDebugReportCallbackEXT debug_report_callback_;
};

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
Runtime* runtime();

// This variable only exists to trigger Runtime initialization upon application
// loading.
static Runtime* runtime_init = runtime();

//
// Impl
//

inline VkInstance Runtime::instance() const {
  return instance_;
}

inline Adapter* Runtime::get_adapter_p() {
  TORCH_CHECK(
      default_adapter_i_ >= 0 && default_adapter_i_ < adapters_.size(),
      "Pytorch Vulkan Runtime: Default device adapter is not set correctly!");
  return &adapters_[default_adapter_i_];
}

inline Adapter& Runtime::get_adapter() {
  TORCH_CHECK(
      default_adapter_i_ >= 0 && default_adapter_i_ < adapters_.size(),
      "Pytorch Vulkan Runtime: Default device adapter is not set correctly!");
  return adapters_[default_adapter_i_];
}

inline Adapter* Runtime::get_adapter_p(size_t i) {
  return &adapters_[i];
}

inline Adapter& Runtime::get_adapter(size_t i) {
  return adapters_[i];
}

inline size_t Runtime::default_adapter_i() const {
  return default_adapter_i_;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
