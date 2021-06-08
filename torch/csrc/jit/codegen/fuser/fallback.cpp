#include <torch/csrc/jit/codegen/fuser/fallback.h>

#include <ATen/core/functional.h> //fmap
#include <ATen/core/stack.h>
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <stdexcept>

namespace torch {
namespace jit {
namespace fuser {

namespace {
c10::AliasAnalysisKind aliasAnalysisIsSpecialCase() {
  return AliasAnalysisKind::INTERNAL_SPECIAL_CASE;
}
} // namespace

// Registers fused operators so that fused graphs can properly generate fallback
// code.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterOperators reg_fused_operators({Operator(
    prim::FusedConcat,
    [](const Node* node) -> Operation {
      int64_t dim = node->i(attr::dim);
      int64_t num_inputs = node->inputs().size();
      return [dim, num_inputs](Stack* stack) {
        auto result = at::cat(
            fmap(
                last(stack, num_inputs),
                [](const IValue& i) { return i.toTensor(); }),
            dim);
        drop(stack, num_inputs);
        pack(stack, std::move(result));
      };
    },
    aliasAnalysisIsSpecialCase())});

void runFallback(int64_t key, Stack& stack) {
  auto maybe_spec = retrieve(key);
  if (!maybe_spec)
    throw std::runtime_error("Failed to find fusion spec to run fallback.");

  InterpreterState{(*maybe_spec)->code()}.run(stack);
}

} // namespace fuser
} // namespace jit
} // namespace torch
