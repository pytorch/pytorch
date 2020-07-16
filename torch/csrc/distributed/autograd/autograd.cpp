#include <torch/csrc/distributed/autograd/autograd.h>
#include <ATen/record_function.h>

namespace torch {
namespace distributed {
namespace autograd {

constexpr auto kDistAutogradBackwardProfilingKey =
    "torch::distributed::autograd::backward";

void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph) {
  RECORD_FUNCTION(
      kDistAutogradBackwardProfilingKey, std::vector<c10::IValue>());
  try {
    DistEngine::getInstance().execute(context_id, roots, retain_graph);
  } catch (std::exception& e) {
    // FIXME: crashes if exception type is not RuntimeError
    throw std::runtime_error(e.what());
  }
}

} // namespace autograd
} // namespace distributed
} // namespace torch
