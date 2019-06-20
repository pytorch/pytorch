#pragma once
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

using Kwargs = std::unordered_map<std::string, IValue>;

// A Function is a pure Graph with no implicit `self` object bound.
// It contains schema information, and the executor that manages the
// execution of the function. script::Method is a wrapper around a
// underlying Function that also provides a `self` object.
struct TORCH_API Function : public std::enable_shared_from_this<Function> {
  Function(
      std::string name,
      bool optimize,
      std::shared_ptr<Graph> graph,
      std::function<void(Function&)> function_creator)
      : name_(std::move(name)),
        graph_(std::move(graph)),
        optimize_(optimize),
        function_creator_(std::move(function_creator)) {}

  void run(Stack& stack) {
    get_executor().run(stack);
  }

  void run(Stack&& stack) {
    run(stack);
  }

  IValue operator()(
      std::vector<IValue> stack,
      const Kwargs& kwargs = Kwargs()) {
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    run(stack);
    return stack.front();
  }

  std::shared_ptr<Graph> graph() const {
    return graph_;
  }

  const std::string& name() const {
    return name_;
  }

  // if this isn't yet defined, run its method_creator function
  void ensure_defined();

  size_t num_inputs() const {
    return graph()->inputs().size();
  }

  Function& setSchema(FunctionSchema schema) {
    schema_ = make_unique<FunctionSchema>(std::move(schema));
    return *this;
  }

  const FunctionSchema& getSchema() const {
    if (schema_ == nullptr) {
      schema_ = make_unique<FunctionSchema>(defaultSchemaFor(*this));
    }
    return *schema_;
  }

  std::string pretty_print_schema() const {
    AT_ASSERT(schema_);
    std::stringstream ss;
    ss << *schema_;
    return ss.str();
  }

  GraphExecutorState getDebugState() {
    return get_executor().getDebugState();
  }

  bool is_optimized() const {
    return optimize_;
  }

  void check_single_output() {
    TORCH_CHECK(
        graph()->outputs().size() == 1,
        "Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs");
  }

  GraphExecutor& get_executor() {
    std::call_once(executor_init_, [&] {
      check_single_output();
      executor_ = GraphExecutor(graph(), optimize_);
    });
    return executor_;
  }

 private:
  static FunctionSchema defaultSchemaFor(const Function& function) {
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

  std::string name_;
  std::shared_ptr<Graph> graph_; // for debugging and for inlining
  bool optimize_;

  GraphExecutor executor_; // for execution

  std::once_flag executor_init_;

  // an optional function that actually creates the method when
  // ensure_defined() is called. This is used by the compiler so
  // that it can construct methods out of order
  std::function<void(Function&)> function_creator_;

  // if absent, then we generate a default schema based on the graph
  // mutable because getSchema caches the default schema if one is requested
  // before a call to setSchema
  mutable std::unique_ptr<FunctionSchema> schema_;
};
} // namespace jit
} // namespace torch
