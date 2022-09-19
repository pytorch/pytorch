#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/QueryPool.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

#ifdef USE_VULKAN_GPU_DIAGNOSTICS

struct ShaderDurationAggregate final {
  uint32_t idx;

  // Execution Properties
  std::string kernel_name;
  VkExtent3D global_workgroup_size;
  VkExtent3D local_workgroup_size;

  // Timings
  float duration_min;
  float duration_max;
  float duration_sum;
  uint64_t duration_count;
};

struct ShaderDurationReport final {
  api::Context* context_;
  std::vector<ShaderDurationAggregate> entries;

  explicit ShaderDurationReport(api::Context* context)
      : context_{context}, entries{} {}

  void initialize();
  void begin_recording_pass();
  void end_recording_pass();
  std::string generate_string_report();
};

#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
