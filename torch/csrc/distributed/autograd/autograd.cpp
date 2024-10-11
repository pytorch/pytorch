#include <ATen/record_function.h>
#include <torch/csrc/distributed/autograd/autograd.h>

namespace torch::distributed::autograd {

constexpr auto kDistAutogradBackwardProfilingKey =
    "torch::distributed::autograd::backward";

void backward(
    int64_t context_id,
    const variable_list& roots,
    bool retain_graph) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.autograd.backward");
  RECORD_FUNCTION(
      kDistAutogradBackwardProfilingKey, std::vector<c10::IValue>());
  try {
    DistEngine::getInstance().execute(context_id, roots, retain_graph);
  } catch (std::exception& e) {
    // FIXME: crashes if exception type is not RuntimeError
    TORCH_CHECK(false, e.what());
  }
}

} // namespace torch::distributed::autograd
