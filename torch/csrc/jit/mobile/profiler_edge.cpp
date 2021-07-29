#include <torch/csrc/jit/mobile/profiler_edge.h>

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
    const bool with_module_hierarchy) :
  m_(m), trace_file_name_(fname) {
  profiler::ProfilerConfig config(
      profiler::ProfilerState::KINETO,
      report_input_shapes,
      profile_memory,
      with_stack,
      with_flops,
      with_module_hierarchy);
  profiler::prepareProfiler(config, {profiler::ActivityType::CPU});
  if (with_module_hierarchy || with_stack) {
    auto post_processing = [this, with_stack, with_module_hierarchy](std::vector<profiler::KinetoEvent>& events) {
      for(auto& e : events) {
        if (with_module_hierarchy) {
          e.moduleHierarchy(this->m_.getModuleHierarchy(e.debugHandle()));
        } else if (with_stack) {
          e.moduleHierarchy(this->m_.getCallStack(e.debugHandle()));
        }
      }
    };
    profiler::enableProfilerWithEventPostProcess(
        config, {profiler::ActivityType::CPU}, post_processing, {at::RecordScope::LITE_INTERPRETER});
  } else {
    profiler::enableProfiler(config, {profiler::ActivityType::CPU}, {at::RecordScope::LITE_INTERPRETER});
  }
  trace_file_name_ = fname;
}

KinetoEdgeCPUProfiler::~KinetoEdgeCPUProfiler() {
  profiler::disableProfiler()->save(trace_file_name_);
}
} // namespace mobile
} // namespace jit
} // namespace torch
