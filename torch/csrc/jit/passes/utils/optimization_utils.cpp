#include <torch/csrc/jit/passes/utils/optimization_utils.h>

namespace torch::jit {

bool nonConstantParameters(Node* n) {
  // Checks if the parameters, not including the
  // first param are all constants.
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

} // namespace torch::jit
