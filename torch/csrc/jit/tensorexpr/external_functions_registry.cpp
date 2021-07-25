#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch {
namespace jit {
namespace tensorexpr {

std::unordered_map<std::string, NNCExternalFunction>& getNNCFunctionRegistry() {
  static std::unordered_map<std::string, NNCExternalFunction> func_registry_;
  return func_registry_;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
