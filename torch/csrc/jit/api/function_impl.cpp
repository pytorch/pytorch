#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/passes/inliner.h>

#include <torch/csrc/jit/frontend/error_report.h>

namespace torch {
namespace jit {

void placeholderCreator(GraphFunction&) {
  throw RecursiveMethodCallError();
}

void GraphFunction::run(Stack& stack) {
  get_executor().run(stack);
}

void GraphFunction::run(Stack&& stack) {
  run(stack);
}

IValue GraphFunction::operator()(
    std::vector<IValue> stack,
    const Kwargs& kwargs) {
  getSchema().checkAndNormalizeInputs(stack, kwargs);
  run(stack);
  return stack.front();
}

void GraphFunction::ensure_defined() {
  if (function_creator_) {
    auto creator = function_creator_;
    function_creator_ = placeholderCreator;
    creator(*this);
    function_creator_ = nullptr;
  }
  check_single_output();
}

void preoptimizeGraph(std::shared_ptr<Graph>& graph) {
  // TODO: Invoke cleanup passes before and after inlining to reduce amount of
  // code we're copying.
  Inline(*graph);
}

} // namespace jit
} // namespace torch
