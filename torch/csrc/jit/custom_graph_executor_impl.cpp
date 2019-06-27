#include <torch/csrc/jit/custom_graph_executor_impl.h>

namespace torch {
namespace jit {

namespace {

struct GraphExecutorImplState {
  Symbol name = kDefaultExecutor;
  bool impl_cached = false;
  GraphExecutorImplCreator impl_fn;
};
thread_local GraphExecutorImplState executor_state;

/* Meyers Singleton */
std::unordered_map<Symbol, GraphExecutorImplCreator>& getGraphExecutorImpls() {
  static std::unordered_map<Symbol, GraphExecutorImplCreator> executors;
  return executors;
}

} // namespace

Symbol getGraphExecutorName() {
  return executor_state.name;
}

void setGraphExecutorName(Symbol name) {
  auto& executors = getGraphExecutorImpls();
  auto it = executors.find(name);
  TORCH_CHECK(
      it != executors.end(),
      "No GraphExecutorImpl registered for ",
      name.toDisplayString());

  executor_state.name = name;
  executor_state.impl_cached = true;
  executor_state.impl_fn = it->second;
}

RegisterGraphExecutorImpl::RegisterGraphExecutorImpl(
    Symbol name,
    GraphExecutorImplCreator creator) {
  auto& executors = getGraphExecutorImpls();
  TORCH_CHECK(
      executors.emplace(name, creator).second,
      "Cannot register GraphExecutorImpl for ",
      name.toDisplayString());
}

GraphExecutorImplCreator getGraphExecutorImpl() {
  if (!executor_state.impl_cached)
    setGraphExecutorName(executor_state.name);
  return executor_state.impl_fn;
}

} // namespace jit
} // namespace torch
