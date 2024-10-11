#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/api/Utils.h>
#ifdef USE_KINETO
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/orchestration/vulkan.h>
#endif // USE_KINETO

#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {
// On Mali gpus timestamp_period seems to return 0.
// For some reason when 52.08 is used op runtimes seem to make more sense
// TODO: Figure out what is special about 52.08
constexpr int64_t kDefaultNsPerTick = 52; // lround(52.08f);
} // namespace

QueryPool::QueryPool(const QueryPoolConfig& config, const Adapter* adapter_p)
    : mutex_{},
      device_(adapter_p->device_handle()),
      config_(config),
      querypool_(VK_NULL_HANDLE),
      shader_logs_(1),
      in_use_(0),
      previous_shader_count_(0u),
      results_pending_(false) {
  const VkQueryPoolCreateInfo info{
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      VK_QUERY_TYPE_TIMESTAMP, // queryType
      config_.maxQueryCount, // queryCount
      0u, // pipelineStatistics
  };

  VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &querypool_));

  shader_log().reserve(config_.initialReserveSize);

  VK_CHECK_COND(adapter_p, "Valid GPU device must be created for QueryPool");
  ns_per_tick_ = std::lround(adapter_p->timestamp_period());
  ns_per_tick_ = (ns_per_tick_ == 0) ? kDefaultNsPerTick : ns_per_tick_;

#ifdef USE_KINETO
  torch::profiler::impl::vulkan::registerGetShaderNameAndDurationNs(
      [this](int64_t vulkan_id) {
        return get_shader_name_and_execution_duration_ns(vulkan_id);
      });
#endif // USE_KINETO
}

QueryPool::~QueryPool() {
  if (VK_NULL_HANDLE == querypool_) {
    return;
  }
  vkDestroyQueryPool(device_, querypool_, nullptr);

#ifdef USE_KINETO
  torch::profiler::impl::vulkan::deregisterGetShaderNameAndDurationNs();
#endif // USE_KINETO
}

void QueryPool::reset(const CommandBuffer& cmd) {
  std::lock_guard<std::mutex> lock(mutex_);
  cmd.reset_querypool(querypool_, 0u, in_use_);
  previous_shader_count_ += shader_log().size();
  in_use_ = 0u;
  shader_logs_.emplace_back();
  shader_log().reserve(config_.initialReserveSize);
  results_pending_ = false;
}

size_t QueryPool::write_timestamp(const CommandBuffer& cmd) {
  VK_CHECK_COND(
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

  uint32_t log_idx = shader_log().size();
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

  shader_log().emplace_back(log_entry);

  results_pending_ = true;

#ifdef USE_KINETO
  torch::profiler::impl::vulkan_id_t vulkan_id =
      torch::profiler::impl::vulkan_id_t(previous_shader_count_ + log_idx);

  torch::profiler::impl::_reportVulkanEventToProfiler(vulkan_id);
#endif // USE_KINETO

  return log_idx;
}

void QueryPool::shader_profile_end(
    const CommandBuffer& cmd,
    const uint32_t log_idx) {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t query_idx = write_timestamp(cmd);

  shader_log()[log_idx].end_query_idx = query_idx;
}

void QueryPool::extract_results() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!results_pending_) {
    return;
  }

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

  for (ShaderDuration& entry : shader_log()) {
    entry.start_time_ns = query_data.at(entry.start_query_idx) * ns_per_tick_;
    entry.end_time_ns = query_data.at(entry.end_query_idx) * ns_per_tick_;
    entry.execution_duration_ns = entry.end_time_ns - entry.start_time_ns;
  }

  results_pending_ = false;
}

static std::string stringize(const VkExtent3D& extents) {
  std::stringstream ss;
  ss << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return ss.str();
}

std::string QueryPool::generate_string_report() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::stringstream ss;

  int kernel_name_w = 40;
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

  for (ShaderDuration& entry : shader_log()) {
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

uint64_t QueryPool::get_total_op_ns(const std::string& op_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t sum = 0;
  for (ShaderDuration& entry : shader_log()) {
    if (entry.kernel_name == op_name) {
      sum += entry.execution_duration_ns;
    }
  }
  return sum;
}

void QueryPool::shader_log_for_each(
    std::function<void(const ShaderDuration&)> fn) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::for_each(shader_log().begin(), shader_log().end(), std::move(fn));
}

std::tuple<std::string, uint64_t> QueryPool::
    get_shader_name_and_execution_duration_ns(size_t query_index) {
  extract_results();

  std::lock_guard<std::mutex> lock(mutex_);

  const size_t entry_count = shader_logs_entry_count_thread_unsafe();
  VK_CHECK_COND(
      (query_index >= 0 && query_index < entry_count),
      "query_index of ",
      query_index,
      " is out of bounds (",
      entry_count,
      ") in QueryPool::get_shader_name_and_duration_ns");

  size_t log_idx = 0;
  size_t entry_count_acc = 0;
  while (entry_count_acc + shader_logs_[log_idx].size() <= query_index) {
    entry_count_acc += shader_logs_[log_idx].size();
    log_idx += 1;
  }

  const ShaderDuration& entry =
      shader_logs_[log_idx][query_index - entry_count_acc];

  return std::tuple<std::string, uint64_t>(
      entry.kernel_name, entry.execution_duration_ns);
}

size_t QueryPool::shader_logs_entry_count_thread_unsafe() {
  return previous_shader_count_ + shader_log().size();
}

size_t QueryPool::shader_logs_entry_count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return shader_logs_entry_count_thread_unsafe();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
