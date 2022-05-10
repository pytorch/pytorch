#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class QueryPool final {
 public:
  explicit QueryPool(const VkDevice& device, const bool is_timestamps_supported, const float timestamp_period_us);
  QueryPool(const QueryPool&) = delete;
  QueryPool(QueryPool&&) = default;
  QueryPool& operator=(const QueryPool&) = delete;
  QueryPool& operator=(QueryPool&&) = default;
  ~QueryPool();

public:
  struct PerfInfo final {
    std::string query_name;
    int64_t start_time_us;
    int64_t end_time_us;
    int64_t execution_time_us;
  };

  struct Configuration final {
    static constexpr uint32_t kTimestampsPerQuery = 2u;
    static constexpr uint32_t kMaxQueryCount = 65536u;
  };

public:
  bool is_enabled() const;
  bool enable();
  std::vector<QueryPool::PerfInfo> disable(const bool waitfor_allqueries = true);
  int begin(const VkCommandBuffer& commandBuffer, const std::string& query_name);
  void end(const VkCommandBuffer& commandBuffer, const int queryIndex);
  std::vector<PerfInfo> result(const bool waitfor_allqueries) const;

private:
  VkDevice device_;
  bool is_timestamps_supported_;
  float timestamp_period_us_;
  VkQueryPool querypool_;
  std::vector<std::string> query_names_;
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
