#include <torch/csrc/jit/python/update_graph_executor_opt.h>

namespace torch::jit {

thread_local bool kOptimize = true;
void setGraphExecutorOptimize(bool o) {
  kOptimize = o;
}
bool getGraphExecutorOptimize() {
  return kOptimize;
}

} // namespace torch::jit
