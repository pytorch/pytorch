#include <torch/csrc/jit/profiling_graph_executor_impl.h>

namespace torch {
namespace jit {

thread_local bool profiling_mode = false;
bool& getProfilingMode() {
  return profiling_mode;
}

void ProfilingGraphExecutorImpl::run(Stack& stack) {
  TORCH_CHECK(
      stack.size() >= num_inputs,
      "expected ",
      num_inputs,
      " inputs, but got only ",
      stack.size());

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
    exec_plan_->run(stack);
  } else {
    AT_ERROR("Not yet implemented");
  }
  return;
}

GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  AT_ERROR("not supported");
}

} // namespace jit
} // namespace torch
