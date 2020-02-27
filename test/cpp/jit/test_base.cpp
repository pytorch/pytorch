#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include "torch/csrc/jit/custom_operator.h"

namespace torch {
namespace jit {
inline c10::OperatorOptions aliasAnalysisFromSchema() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return result;
}

RegisterOperators reg({
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
