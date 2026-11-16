#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h>

namespace torch::jit {

static thread_local bool kOptimize = true;
void setGraphExecutorOptimize(bool o) {
  kOptimize = o;
  GRAPH_DEBUG("GraphExecutorOptimize set to ", o);
}
bool getGraphExecutorOptimize() {
  return kOptimize;
}

} // namespace torch::jit
