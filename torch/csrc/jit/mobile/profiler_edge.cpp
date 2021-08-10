#include <torch/csrc/jit/mobile/profiler_edge.h>
#include <string>
#include <vector>

namespace profiler = torch::autograd::profiler;
namespace torch {
namespace jit {
namespace mobile {

KinetoEdgeCPUProfiler::KinetoEdgeCPUProfiler(
    const torch::jit::mobile::Module& m,
    const std::string& fname,
    const bool report_input_shapes,
    const bool profile_memory,
    const bool with_stack,
    const bool with_flops,
    const bool with_modules)
    : m_(m), trace_file_name_(fname) {
  profiler::ProfilerConfig config(
      profiler::ProfilerState::KINETO,
      report_input_shapes,
      profile_memory,
      with_stack,
      with_flops,
      with_modules);
  profiler::prepareProfiler(config, {profiler::ActivityType::CPU});
  if (with_modules || with_stack) {
    auto post_processing = [this, with_stack, with_modules](
                               std::vector<profiler::KinetoEvent>& events) {
      for (auto& e : events) {
        if (with_modules) {
          // Since KinetoEvents's module hierarchy takes vector of strings we
          // just construct a temporary vector using one string element
          e.moduleHierarchy(std::vector<std::string>(
              {this->m_.getModuleHierarchy(e.debugHandle())}));
        } else if (with_stack) {
          // Since KinetoEvents's stack trace takes vector of strings we just
          // construct a temporary vector using one string element
          e.stack(std::vector<std::string>(
              {this->m_.getCallStack(e.debugHandle())}));
        }
      }
    };
    profiler::enableProfilerWithEventPostProcess(
        config,
        {profiler::ActivityType::CPU},
        post_processing,
        {at::RecordScope::LITE_INTERPRETER});
  } else {
    profiler::enableProfiler(
        config,
        {profiler::ActivityType::CPU},
        {at::RecordScope::LITE_INTERPRETER});
  }
  trace_file_name_ = fname;
}

KinetoEdgeCPUProfiler::~KinetoEdgeCPUProfiler() {
  profiler::disableProfiler()->save(trace_file_name_);
}
} // namespace mobile
} // namespace jit
} // namespace torch
