#include <torch/csrc/jit/custom_graph_executor_impl.h>

namespace torch {
namespace jit {

thread_local Symbol executor_key = kDefaultExecutor;
Symbol& getGraphExecutorName() {
  return executor_key;
}

/* Meyers Singleton */
std::unordered_map<Symbol, GraphExecutorImplCreator>& getGraphExecutorImpls() {
  static std::unordered_map<Symbol, GraphExecutorImplCreator> executors;
  return executors;
};

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
  auto& executors = getGraphExecutorImpls();
  auto it = executors.find(executor_key);

  TORCH_CHECK(
      it != executors.end(),
      "No GraphExecutorImpl registered for ",
      executor_key.toDisplayString());
  return it->second;
}

} // namespace jit
} // namespace torch
