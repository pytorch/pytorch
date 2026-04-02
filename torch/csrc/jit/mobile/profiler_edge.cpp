#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/jit/mobile/profiler_edge.h>
#include <string>
#include <vector>

namespace torch::jit::mobile {

thread_local KinetoEdgeCPUProfiler* tls_edge_profiler{nullptr};

KinetoEdgeCPUProfiler::KinetoEdgeCPUProfiler(
    const torch::jit::mobile::Module& m,
    const std::string& fname,
    const bool report_input_shapes,
    const bool profile_memory,
    const bool with_stack,
    const bool with_flops,
    const bool with_modules,
    std::vector<std::string> events,
    const bool adjust_vulkan_timestamps)
    : m_(m), trace_file_name_(fname) {
  torch::profiler::impl::ExperimentalConfig experimental_config;
  // Enable hardware counters
  if (!events.empty()) {
    experimental_config.performance_events = std::move(events);
  }

  // Adjust vulkan timestamps from query pool to align with cpu event times
  experimental_config.adjust_timestamps = adjust_vulkan_timestamps;

  torch::profiler::impl::ProfilerConfig config(
      torch::profiler::impl::ProfilerState::KINETO,
      report_input_shapes,
      profile_memory,
      with_stack,
      with_flops,
      with_modules,
      experimental_config);
  torch::autograd::profiler::prepareProfiler(
      config, {torch::autograd::profiler::ActivityType::CPU});
  if (with_modules || with_stack) {
    auto post_processing = [this, with_stack, with_modules](
                               int64_t debug_handle,
                               std::vector<std::string>& jit_stack,
                               std::vector<std::string>& jit_modules) {
      std::string no_debug_info("Model was not saved with debug information");
      if (with_modules) {
        // Since KinetoEvents's module hierarchy takes vector of strings
        // we just construct a temporary vector using one string element
        jit_modules = std::vector<std::string>(
            {this->m_.hasDebugHandles()
                 ? this->m_.getModuleHierarchy(debug_handle)
                 : no_debug_info});
      } else if (with_stack) {
        // Since KinetoEvents's stack trace takes vector of strings we
        // just construct a temporary vector using one string element
        jit_stack = std::vector<std::string>(
            {this->m_.hasDebugHandles() ? this->m_.getCallStack(debug_handle)
                                        : no_debug_info});
      }
    };
    torch::autograd::profiler::enableProfilerWithEventPostProcess(
        config,
        {torch::autograd::profiler::ActivityType::CPU},
        post_processing,
        {at::RecordScope::LITE_INTERPRETER});
  } else {
    torch::autograd::profiler::enableProfiler(
        config,
        {torch::autograd::profiler::ActivityType::CPU},
        {at::RecordScope::LITE_INTERPRETER});
  }
  trace_file_name_ = fname;
  TORCH_CHECK(
      tls_edge_profiler == nullptr, "Edge profiler is already profiling.")
  tls_edge_profiler = this;
}

void KinetoEdgeCPUProfiler::recordBackendMemoryEvent(
    void* ptr,
    int64_t alloc_size,
    size_t total_allocated,
    size_t total_reserved,
    c10::Device device) {
  c10::reportMemoryUsageToProfiler(
      ptr, alloc_size, total_allocated, total_reserved, device);
}

void KinetoEdgeCPUProfiler::recordBackendEvent(
    const int64_t start_time_us,
    const int64_t end_time_us,
    const int64_t debug_handle,
    const std::string& event_name,
    const std::string& backend_name) {
  torch::autograd::profiler::reportBackendEventToActiveKinetoProfiler(
      start_time_us,
      end_time_us,
      debug_handle,
      at::RecordScope::LITE_INTERPRETER,
      event_name,
      backend_name);
}

const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
KinetoEdgeCPUProfiler::disableProfiler() {
  TORCH_CHECK(
      !profiler_result_,
      "KinetoEdgeCPUProfiler already disabled. "
      "To get list of events use getProfilerResults()");
  profiler_result_ = torch::autograd::profiler::disableProfiler();
  return profiler_result_;
}

const std::unique_ptr<torch::autograd::profiler::ProfilerResult>&
KinetoEdgeCPUProfiler::getProfilerResult() {
  TORCH_CHECK(
      profiler_result_,
      "KinetoEdgeCPUProfiler has not been disabled. "
      "use disableProfiler() API first, which returns the ProfilerResult.");
  return profiler_result_;
}

KinetoEdgeCPUProfiler::~KinetoEdgeCPUProfiler() {
  if (!trace_file_name_.empty()) {
    if (profiler_result_) {
      profiler_result_->save(trace_file_name_);
    } else {
      torch::autograd::profiler::disableProfiler()->save(trace_file_name_);
    }
  }
  tls_edge_profiler = nullptr;
}

KinetoEdgeCPUProfiler* getCurrentEdgeProfiler() {
  return tls_edge_profiler;
}

} // namespace torch::jit::mobile
