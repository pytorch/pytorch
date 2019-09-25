#pragma once
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/utils/memory.h>
#include <mutex>

namespace torch {
namespace jit {

using Kwargs = std::unordered_map<std::string, IValue>;

TORCH_API void preoptimizeGraph(std::shared_ptr<Graph>& graph);

// A Function is a pure Graph with no implicit `self` object bound.
// It contains schema information, and the executor that manages the
// execution of the function. script::Method is a wrapper around a
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  Function(
      c10::QualifiedName name,
      std::shared_ptr<Graph> graph,
      std::function<void(Function&)> function_creator)
      : name_(std::move(name)),
        graph_(std::move(graph)),
        function_creator_(std::move(function_creator)) {}

  void run(Stack& stack);

  void run(Stack&& stack);

  IValue operator()(
      std::vector<IValue> stack,
      const Kwargs& kwargs = Kwargs());

  std::shared_ptr<Graph> graph() const {
    return graph_;
  }

  std::shared_ptr<Graph> optimized_graph() const {
    if (optimized_graph_) {
      return *optimized_graph_;
    }
    std::lock_guard<std::mutex> lock(compile_mutex);
    optimized_graph_ = graph_->copy();
    preoptimizeGraph(*optimized_graph_);
    return *optimized_graph_;
  }

  const c10::QualifiedName& qualname() const {
    return name_;
  }

  const std::string& name() const {
    return name_.name();
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

  const FunctionSchema& getSchema() const;

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
    AT_WARN(
        "Function::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
    return true;
  }

  void check_single_output() {
    TORCH_CHECK(
        graph()->outputs().size() == 1,
        "Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs");
  }

  GraphExecutor& get_executor() {
    ensure_defined();
    if (executor_) {
      return executor_;
    }
    std::lock_guard<std::mutex> lock(compile_mutex);
    check_single_output();
    executor_ = GraphExecutor(graph());
    return executor_;
  }

 private:
  c10::QualifiedName name_;
  // The original, non-optimized graph
  std::shared_ptr<Graph> graph_; // for debugging and for inlining

  // Optimized graph, computed lazily. Used for inlining.
  // Note: this graph is not specialized, only generic optimizations are applied
  // here.
  mutable c10::optional<std::shared_ptr<Graph>> optimized_graph_;

  // Functions are invokable from multiple threads, so this lock needs to be
  // held when we're initializing graph executor for the first time or computing
  // the optimized graph.
  mutable std::mutex compile_mutex;

  GraphExecutor executor_; // for execution

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
