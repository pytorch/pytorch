#include <ATen/native/vulkan/api/ShaderReporting.h>
#include <ATen/native/vulkan/api/Utils.h>

#include <iostream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

#ifdef USE_VULKAN_GPU_DIAGNOSTICS

void ShaderDurationReport::initialize() {
  context_->querypool().extract_results();

  std::vector<ShaderDurationAggregate> log;
  int i = 0;
  for (const api::ShaderDuration& log_entry :
       context_->querypool().shader_log()) {
    TORCH_CHECK(i == log_entry.idx);
    ShaderDurationAggregate report_entry{
        log_entry.idx,
        log_entry.kernel_name,
        log_entry.global_workgroup_size,
        log_entry.local_workgroup_size,
        log_entry.execution_duration_us,
        log_entry.execution_duration_us,
        log_entry.execution_duration_us,
        1u,
    };
    entries.emplace_back(report_entry);

    ++i;
  }
}

void ShaderDurationReport::begin_recording_pass() {
  context_->reset_querypool();
}

void ShaderDurationReport::end_recording_pass() {
  context_->querypool().extract_results();
  for (const api::ShaderDuration& log_entry :
       context_->querypool().shader_log()) {
    ShaderDurationAggregate& report_entry = entries[log_entry.idx];

    TORCH_CHECK(log_entry.kernel_name == report_entry.kernel_name);

    report_entry.duration_min =
        std::min(report_entry.duration_min, log_entry.execution_duration_us);
    report_entry.duration_max =
        std::max(report_entry.duration_max, log_entry.execution_duration_us);
    report_entry.duration_sum += log_entry.execution_duration_us;
    report_entry.duration_count += 1;
  }
}

std::string ShaderDurationReport::generate_string_report() {
  std::stringstream ss;

  int kernel_name_w = 25;
  int global_size_w = 20;
  int local_size_w = 20;
  int duration_min_w = 25;
  int duration_max_w = 20;
  int duration_avg_w = 20;

  ss << std::fixed << std::setprecision(2);
  ss << std::left;
  ss << std::setw(kernel_name_w) << "Kernel Name";
  ss << std::setw(global_size_w) << "Global Size";
  ss << std::setw(local_size_w) << "Local Size";
  ss << std::right << std::setw(duration_min_w) << "Dur Min (ns)";
  ss << std::right << std::setw(duration_max_w) << "Dur Max (ns)";
  ss << std::right << std::setw(duration_avg_w) << "Dur Avg (ns)";
  ss << std::endl;

  ss << std::left;
  ss << std::setw(kernel_name_w) << "===========";
  ss << std::setw(global_size_w) << "===========";
  ss << std::setw(local_size_w) << "==========";
  ss << std::right << std::setw(duration_min_w) << "============";
  ss << std::right << std::setw(duration_max_w) << "============";
  ss << std::right << std::setw(duration_avg_w) << "============";
  ss << std::endl;

  long total_duration_us = 0u;

  for (ShaderDurationAggregate& entry : entries) {
    float duration_avg_us = entry.duration_sum / entry.duration_count;

    ss << std::left;
    ss << std::setw(kernel_name_w) << entry.kernel_name;
    ss << std::setw(global_size_w) << stringize(entry.global_workgroup_size);
    ss << std::setw(local_size_w) << stringize(entry.local_workgroup_size);
    ss << std::right << std::setw(duration_min_w) << entry.duration_min;
    ss << std::right << std::setw(duration_max_w) << entry.duration_max;
    ss << std::right << std::setw(duration_avg_w) << duration_avg_us;
    ss << std::endl;

    total_duration_us += duration_avg_us;
  }

  int dur_label_w = 20;
  int total_dur_w = 20;
  ss << std::endl;
  ss << std::left << std::setw(dur_label_w) << "Total Duration us: ";
  ss << std::right << std::setw(total_dur_w) << total_duration_us;
  ss << std::endl;

  return ss.str();
}

#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
