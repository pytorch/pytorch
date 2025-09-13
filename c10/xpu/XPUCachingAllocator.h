#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/xpu/XPUStream.h>
#include <c10/util/env.h>

namespace c10::xpu::XPUCachingAllocator {

// Environment config parser
class C10_XPU_API XPUAllocatorConfig {
 public:
  static bool expandable_segments() {
#ifndef SYCL_EXT_ONEAPI_VIRTUAL_MEM
    if (instance().m_expandable_segments) {
      TORCH_WARN_ONCE("expandable_segments not supported on this platform")
    }
    return false;
#else
    return instance().m_expandable_segments;
#endif
  }

  static std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(
        instance().m_last_allocator_settings_mutex);
    return instance().m_last_allocator_settings;
  }

  static XPUAllocatorConfig& instance() {
    static XPUAllocatorConfig* s_instance = ([]() {
      auto inst = new XPUAllocatorConfig();
      auto env = c10::utils::get_env("PYTORCH_XPU_ALLOC_CONF");
      inst->parseArgs(env);
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const std::optional<std::string>& env);

 private:
  XPUAllocatorConfig();

  static void lexArgs(const std::string& env, std::vector<std::string>& config);
  static void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);

  std::atomic<bool> m_expandable_segments;
  std::string m_last_allocator_settings;
  std::mutex m_last_allocator_settings_mutex;
};


C10_XPU_API Allocator* get();

C10_XPU_API void init(DeviceIndex device_count);

C10_XPU_API void emptyCache();

C10_XPU_API void resetPeakStats(DeviceIndex device);

C10_XPU_API void resetAccumulatedStats(DeviceIndex device);

C10_XPU_API c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    DeviceIndex device);

C10_XPU_API void* raw_alloc(size_t size);

C10_XPU_API void raw_delete(void* ptr);

C10_XPU_API void recordStream(const DataPtr& dataPtr, XPUStream stream);

// General caching allocator utilities
C10_XPU_API void setAllocatorSettings(const std::string& env);

} // namespace c10::xpu::XPUCachingAllocator
