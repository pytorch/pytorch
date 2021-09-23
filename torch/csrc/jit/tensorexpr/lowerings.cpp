#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

namespace torch {
namespace jit {
namespace tensorexpr {

std::unordered_map<std::string, NNCLoweringFunction>& getNNCLoweringRegistry() {
  static std::unordered_map<std::string, NNCLoweringFunction>
      lowering_registry_;
  return lowering_registry_;
}

NNCLoweringFunction getStandardLoweringFor(const std::string& op) {
  const auto& lowerings = getNNCLoweringRegistry();
  if (lowerings.count(op))
    return lowerings.at(op);
  return nullptr;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
