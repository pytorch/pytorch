#include <torch/csrc/jit/passes/cuda_graph_fuser.h>
#include <mutex>

namespace torch {
namespace jit {

static CudaFuserComparisonCallback comparison_callback = {false, nullptr};
static std::mutex comparison_callback_lock;

CudaFuserComparisonCallback getCudaFuserComparisonCallback() {
  std::lock_guard<std::mutex> guard(comparison_callback_lock);
  return comparison_callback;
}

void setCudaFuserComparisonCallback(CudaFuserComparisonCallback callback) {
  std::lock_guard<std::mutex> guard(comparison_callback_lock);
  comparison_callback = callback;
}

} // namespace jit
} // namespace torch
