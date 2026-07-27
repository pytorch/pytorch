#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch::jit::tensorexpr {

std::unordered_map<std::string, NNCExternalFunction>& getNNCFunctionRegistry() {
  static std::unordered_map<std::string, NNCExternalFunction> func_registry_;
  return func_registry_;
}

} // namespace torch::jit::tensorexpr
