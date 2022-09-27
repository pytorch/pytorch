#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Pipeline.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct QueryPoolConfig final {
  uint32_t maxQueryCount;
  uint32_t initialReserveSize;
};

struct ShaderDuration final {
  uint32_t idx;

  // Execution Properties
  std::string kernel_name;
  VkExtent3D global_workgroup_size;
  VkExtent3D local_workgroup_size;

  // Query indexes
  uint32_t start_query_idx;
  uint32_t end_query_idx;

  // Timings
  uint64_t start_time_ns;
  uint64_t end_time_ns;
  uint64_t execution_duration_ns;
};

class QueryPool final {
 public:
  explicit QueryPool(const QueryPoolConfig&, const Adapter* adapter_p);

  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;

  QueryPool(QueryPool&&) = delete;
  QueryPool& operator=(QueryPool&&) = delete;

  ~QueryPool();

 private:
  std::mutex mutex_;

  VkDevice device_;
  QueryPoolConfig config_;

  VkQueryPool querypool_;

  std::vector<ShaderDuration> shader_log_;
  size_t in_use_;

 private:
  size_t write_timestamp(const CommandBuffer&);

  std::string generate_string_report();

 public:
  inline bool is_enabled() const {
    return VK_NULL_HANDLE != querypool_;
  }

  void reset(const CommandBuffer&);

  uint32_t shader_profile_begin(
      const CommandBuffer&,
      const std::string&,
      const VkExtent3D,
      const VkExtent3D);
  void shader_profile_end(const CommandBuffer&, const uint32_t);

  void extract_results();
  void print_results();
  uint64_t get_total_op_ns(std::string op_name);
  uint64_t ns_per_tick_;
  void shader_log_for_each(std::function<void(const ShaderDuration&)> fn);
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
