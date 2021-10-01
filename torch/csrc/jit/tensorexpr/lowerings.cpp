#include <torch/csrc/jit/tensorexpr/lowerings.h>

namespace torch {
namespace jit {
namespace tensorexpr {

std::unordered_map<std::string, NNCLoweringFunction>& getNNCLoweringRegistry() {
  static std::unordered_map<std::string, NNCLoweringFunction>
      lowering_registry_;
  return lowering_registry_;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
