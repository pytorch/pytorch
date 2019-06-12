#include "torch/csrc/jit/fuser/fallback.h"

#include "torch/csrc/utils/functional.h" //fmap
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/fuser/kernel_cache.h"

#include <stdexcept>

namespace torch { namespace jit { namespace fuser {

// Registers fused operators so that fused graphs can properly generate fallback code.
RegisterOperators reg_fused_operators({
  Operator(
    prim::FusedConcat
  , [](const Node* node) {
    int64_t dim = node->i(attr::dim);
    int64_t num_inputs = node->inputs().size();
    return [dim, num_inputs](Stack& stack) {
    auto result = at::cat(
      fmap(last(stack, num_inputs), [](const IValue& i) { return i.toTensor(); })
    , dim
    );
    drop(stack, num_inputs);
    pack(stack, std::move(result));
    return 0;
    };
  })
});

void runFallback(int64_t key, Stack& stack) {
  auto maybe_spec = retrieve(key);
  if (!maybe_spec)
    throw std::runtime_error("Failed to find fusion spec to run fallback.");
  
  InterpreterState{(*maybe_spec)->code()}.run(stack);
}

} // namespace fuser
} // namespace jit
} // namespace torch
