#include <torch/csrc/jit/python/update_graph_executor_opt.h>

namespace torch {
namespace jit {

thread_local bool kOptimize = true;
void setGraphExecutorOptimize(bool o) {
  kOptimize = o;
}
bool getGraphExecutorOptimize() {
  return kOptimize;
}
} // namespace jit
} // namespace torch
