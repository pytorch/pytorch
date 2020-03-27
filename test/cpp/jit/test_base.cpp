#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include "torch/csrc/jit/runtime/custom_operator.h"

namespace torch {
namespace jit {
inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators reg({
    // This operator is intended to be used in JIT analysis and transformation
    // pass unit tests in which Values with type Tensor are often required. It
    // should not be used in situations in which the graph is actually executed
    // because it always produces empty Tensors.
    Operator(
        "prim::MakeTestTensor() -> Tensor",
        [](Stack& stack) {
          push(stack, at::Tensor());
          return 0;
        },
        aliasAnalysisFromSchema()),
});

} // namespace jit
} // namespace torch
