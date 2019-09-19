#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/peephole.h>

#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {
namespace {
FunctionSchema defaultSchemaFor(const Function& function) {
  std::vector<Argument> args;
  std::vector<Argument> returns;
  Graph& g = *function.graph();
  size_t num_inputs = function.num_inputs();
  for (size_t i = 0; i < num_inputs; ++i) {
    const Value* v = g.inputs().at(i);
    std::string name = v->hasDebugName() ? v->debugNameBase()
                                         : ("argument_" + std::to_string(i));
    args.emplace_back(std::move(name), unshapedType(g.inputs()[i]->type()));
  }
  for (size_t i = 0; i < g.outputs().size(); ++i) {
    returns.emplace_back("", unshapedType(g.outputs()[i]->type()));
  }
  return {function.name(), "", std::move(args), std::move(returns)};
}
} // namespace

struct RecursiveMethodCallError : public std::exception {};
void placeholderCreator(Function&) {
  throw RecursiveMethodCallError();
}

void Function::ensure_defined() {
  try {
    if (function_creator_) {
      auto creator = function_creator_;
      function_creator_ = placeholderCreator;
      creator(*this);
      function_creator_ = nullptr;
    }
  } catch (RecursiveMethodCallError&) {
    throw script::ErrorReport() // TODO: once lower_first_class methods is
                                // removed re-establish callsite info for
                                // debugging
        << " method '" << name() << "' is called recursively. "
        << "Recursive calls are not supported";
  }
  check_single_output();
}

const FunctionSchema& Function::getSchema() const {
  if (schema_ == nullptr) {
    schema_ = make_unique<FunctionSchema>(defaultSchemaFor(*this));
  }
  return *schema_;
}

void preoptimizeGraph(std::shared_ptr<Graph>& graph) {
  EliminateDeadCode(graph);
  EliminateCommonSubexpression(graph);
  ConstantPooling(graph);
  PeepholeOptimize(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);

  Inline(*graph);

  EliminateDeadCode(graph);
  EliminateCommonSubexpression(graph);
  ConstantPooling(graph);
  PeepholeOptimize(graph);
  ConstantPropagation(graph);
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
