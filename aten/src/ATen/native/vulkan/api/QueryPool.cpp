#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/ops/Tensor.h>

#include <cmath>
#include <iostream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {
// On Mali gpus timestamp_period seems to return 0.
// For some reason when 52.08 is used op runtimes seem to make more sense
// TODO: Figure out what is special about 52.08
constexpr int64_t default_ns_per_tick = 52; // lround(52.08f);
} // namespace

QueryPool::QueryPool(const QueryPoolConfig& config, const Adapter* adapter_p)
    : mutex_{},
      device_(adapter_p->device_handle()),
      config_(config),
      querypool_(VK_NULL_HANDLE),
      shader_log_{},
      in_use_(0u) {
  const VkQueryPoolCreateInfo info{
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_QUERY_TYPE_TIMESTAMP, // queryType
      config_.maxQueryCount, // queryCount
      0u, // pipelineStatistics
  };

  VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &querypool_));

  shader_log_.reserve(config_.initialReserveSize);

  TORCH_CHECK(adapter_p, "Valid GPU device must be created for QueryPool");
  ns_per_tick_ = std::lround(adapter_p->timestamp_period());
  ns_per_tick_ = (ns_per_tick_ == 0) ? default_ns_per_tick : ns_per_tick_;
}

QueryPool::~QueryPool() {
  if (VK_NULL_HANDLE == querypool_) {
    return;
  }
  vkDestroyQueryPool(device_, querypool_, nullptr);
  shader_log_.clear();
}

void QueryPool::reset(const CommandBuffer& cmd) {
  std::lock_guard<std::mutex> lock(mutex_);
  cmd.reset_querypool(querypool_, 0u, in_use_);
  in_use_ = 0u;
  shader_log_.clear();
}

size_t QueryPool::write_timestamp(const CommandBuffer& cmd) {
  TORCH_CHECK(
      in_use_ < config_.maxQueryCount,
      "Vulkan QueryPool: Exceeded the maximum number of queries "
      "allowed by the queryPool (",
      config_.maxQueryCount,
      ")!");

  cmd.write_timestamp(querypool_, in_use_);

  return in_use_++;
}

uint32_t QueryPool::shader_profile_begin(
    const CommandBuffer& cmd,
    const std::string& kernel_name,
    const VkExtent3D global_workgroup_size,
    const VkExtent3D local_workgroup_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  uint32_t query_idx = write_timestamp(cmd);

  uint32_t log_idx = shader_log_.size();
  ShaderDuration log_entry{
      log_idx,
      // Execution Properties
      kernel_name,
      global_workgroup_size,
      local_workgroup_size,
      // Query indexes
      query_idx, // start query idx
      UINT32_MAX, // end query idx
      // Timings
      0u, // start time
      0u, // end time
      0u, // duration
  };

  shader_log_.emplace_back(log_entry);

  return log_idx;
}

void QueryPool::shader_profile_end(
    const CommandBuffer& cmd,
    const uint32_t log_idx) {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t query_idx = write_timestamp(cmd);

  shader_log_[log_idx].end_query_idx = query_idx;
}

void QueryPool::extract_results() {
  std::lock_guard<std::mutex> lock(mutex_);

  const VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT;

  std::vector<uint64_t> query_data;
  query_data.resize(in_use_);

  VK_CHECK(vkGetQueryPoolResults(
      device_,
      querypool_,
      0u, // firstQuery
      in_use_, // queryCount
      sizeof(uint64_t) * in_use_, // dataSize
      query_data.data(), // pData
      sizeof(uint64_t), // stride
      flags)); // flags

  for (ShaderDuration& entry : shader_log_) {
    entry.start_time_ns = query_data.at(entry.start_query_idx) * ns_per_tick_;
    entry.end_time_ns = query_data.at(entry.end_query_idx) * ns_per_tick_;
    entry.execution_duration_ns = entry.end_time_ns - entry.start_time_ns;
  }
}

std::ostream& operator<<(std::ostream& os, const VkExtent3D& extents) {
  os << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return os;
}

std::string stringize(const VkExtent3D& extents) {
  std::stringstream ss;
  ss << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return ss.str();
}

std::string QueryPool::generate_string_report() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::stringstream ss;

  int kernel_name_w = 25;
  int global_size_w = 15;
  int duration_w = 25;

  ss << std::left;
  ss << std::setw(kernel_name_w) << "Kernel Name";
  ss << std::setw(global_size_w) << "Workgroup Size";
  ss << std::right << std::setw(duration_w) << "Duration (ns)";
  ss << std::endl;

  ss << std::left;
  ss << std::setw(kernel_name_w) << "===========";
  ss << std::setw(global_size_w) << "==============";
  ss << std::right << std::setw(duration_w) << "===========";
  ss << std::endl;

  for (ShaderDuration& entry : shader_log_) {
    std::chrono::duration<size_t, std::nano> exec_duration_ns(
        entry.execution_duration_ns);

    ss << std::left;
    ss << std::setw(kernel_name_w) << entry.kernel_name;
    ss << std::setw(global_size_w) << stringize(entry.global_workgroup_size);
    ss << std::right << std::setw(duration_w) << exec_duration_ns.count();
    ss << std::endl;
  }

  return ss.str();
}

void QueryPool::print_results() {
  std::cout << generate_string_report() << std::endl;
}

uint64_t QueryPool::get_total_op_ns(std::string op_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t sum = 0;
  for (ShaderDuration& entry : shader_log_) {
    if (entry.kernel_name == op_name) {
      sum += entry.execution_duration_ns;
    }
  }
  return sum;
}

void QueryPool::shader_log_for_each(
    std::function<void(const ShaderDuration&)> fn) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::for_each(shader_log_.begin(), shader_log_.end(), fn);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
