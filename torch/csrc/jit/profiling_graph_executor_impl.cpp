#include <torch/csrc/jit/profiling_graph_executor_impl.h>

namespace torch {
namespace jit {

thread_local bool profiling_mode = false;
bool& getProfilingMode() {
  return profiling_mode;
}

ExecutionPlan ProfilingGraphExecutorImpl::getPlanFor(Stack& stack) {
  {
    std::lock_guard<std::mutex> lock(compile_mutex);
    if (!pr_) {
      auto g = graph->copy();
      runRequiredPasses(g);
      pr_ = ProfilingRecord::instrumentGraph(g);
      exec_plan_ = caffe2::make_unique<ExecutionPlan>(pr_->profiled_graph_);
    }
  }
  if (pr_->profiling_count_ > 0) {
    return *exec_plan_;
  }
  AT_ERROR("Not yet implemented");
}

GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  AT_ERROR("not supported");
}

} // namespace jit
} // namespace torch
