#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/ops/Tensor.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VkQueryPool create_query_pool(const VkDevice& device, const uint32_t queryCount) {
  VkQueryPool queryPool{};
  VkQueryPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = queryCount;
  VK_CHECK(vkCreateQueryPool(device, &info, nullptr, &queryPool));
  return queryPool;
};

void destroy_query_pool(const VkDevice& device, const VkQueryPool& querypool) {
  if (VK_NULL_HANDLE != device && VK_NULL_HANDLE != querypool) {
    vkDestroyQueryPool(device, querypool, nullptr);
  }
}

} // namespace

QueryPool::QueryPool(const VkDevice& device, const bool is_timestamps_supported, const float timestamp_period_us)
  : device_(device),
    is_timestamps_supported_(is_timestamps_supported),
    timestamp_period_us_(timestamp_period_us),
    querypool_(VK_NULL_HANDLE) {
}

QueryPool::~QueryPool() {
  destroy_query_pool(device_, querypool_);
  querypool_ = VK_NULL_HANDLE;
  query_names_.clear();
}

bool QueryPool::is_enabled() const {
  return VK_NULL_HANDLE != querypool_;
}

bool QueryPool::enable() {
  TORCH_CHECK(VK_NULL_HANDLE == querypool_, "The query pool already exists.");
  TORCH_CHECK(is_timestamps_supported_, "The device doesn't support for timestamps on all graphics and compute queues.");
  querypool_ = create_query_pool(device_, Configuration::kMaxQueryCount);
  return is_enabled();
}

std::vector<QueryPool::PerfInfo> QueryPool::disable(const bool waitfor_allqueries/* = true*/) {
  auto out = result(waitfor_allqueries);
  destroy_query_pool(device_, querypool_);
  querypool_ = VK_NULL_HANDLE;
  query_names_.clear();
  return out;
}

int QueryPool::begin(const VkCommandBuffer& commandBuffer, const std::string& query_name) {
  if (VK_NULL_HANDLE == querypool_ || VK_NULL_HANDLE == commandBuffer) {
    return -1;
  }
  auto newQueryIndex = static_cast<uint32_t>(query_names_.size());
  TORCH_CHECK(newQueryIndex < Configuration::kMaxQueryCount, "The query index cannot exceed Configuration::kMaxQueryCount.");
  query_names_.push_back(query_name);

  vkCmdWriteTimestamp(
        commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, querypool_, newQueryIndex * Configuration::kTimestampsPerQuery);
  return static_cast<int>(newQueryIndex);
}

void QueryPool::end(const VkCommandBuffer& commandBuffer, const int queryIndex) {
  if (VK_NULL_HANDLE == querypool_ || VK_NULL_HANDLE == commandBuffer) {
    return;
  }
  vkCmdWriteTimestamp(
        commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, querypool_, static_cast<uint32_t>(queryIndex) * Configuration::kTimestampsPerQuery + 1u);
}

std::vector<QueryPool::PerfInfo> QueryPool::result(const bool waitfor_allqueries) const {
  if (VK_NULL_HANDLE == querypool_) {
    return std::vector<QueryPool::PerfInfo> {};
  }

  std::vector<QueryPool::PerfInfo> perfInfo;
  const VkQueryResultFlags flags = waitfor_allqueries ? (VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) : VK_QUERY_RESULT_64_BIT;
  std::array<uint64_t, 2> counter_data{};
  for (uint32_t queryIndex = 0u; queryIndex < query_names_.size(); ++queryIndex) {
    const auto& query_name = query_names_[queryIndex];

    // Grab the gpu timings (nanoseconds)
    auto ret = vkGetQueryPoolResults(device_, querypool_, queryIndex * Configuration::kTimestampsPerQuery, Configuration::kTimestampsPerQuery,
        sizeof(uint64_t) * counter_data.size(), counter_data.data(), sizeof(uint64_t),
        flags);
    if (ret != VK_SUCCESS) {
      std::stringstream msg;
      msg << "vkGetQueryPoolResults() for \"" << query_name << "\"" << " returned an error code " << ret << ".";
      TORCH_WARN(msg.str());
      continue;
    }

    // Tally up GPU time
    int64_t gpu_time_us = static_cast<int64_t>(
        (static_cast<double>(counter_data[1] - counter_data[0]) *
            timestamp_period_us_) / 1'000.f);    // convert ns to us

    perfInfo.emplace_back(QueryPool::PerfInfo {
        query_name,
        static_cast<int64_t>(static_cast<double>(counter_data[0]) * timestamp_period_us_ / 1'000.f),
        static_cast<int64_t>(static_cast<double>(counter_data[1]) * timestamp_period_us_ / 1'000.f),
        gpu_time_us });
 }
  return perfInfo;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
