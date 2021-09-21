#pragma once

#include <ATen/core/function.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

struct TORCH_API GraphFunction : public Function {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  GraphFunction(
      c10::QualifiedName name,
      std::shared_ptr<Graph> graph,
      std::function<void(GraphFunction&)> function_creator)
      : name_(std::move(name)),
        graph_(std::move(graph)),
        function_creator_(std::move(function_creator)) {}

  bool isGraphFunction() const override {
    return true;
  }

  void run(Stack& stack) override;

  void run(Stack&& stack) override;

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch) override;

  IValue operator()(std::vector<IValue> stack, const Kwargs& kwargs = Kwargs())
      override;

  std::shared_ptr<Graph> graph() const override {
    return graph_;
  }

  std::shared_ptr<Graph> optimized_graph() const override {
    std::lock_guard<std::recursive_mutex> lock(compile_mutex);
    auto& optimized_graph = optimized_graphs_[currentSpecialization()];
    if (optimized_graph) {
      return *optimized_graph;
    }
    optimized_graph = graph_->copy();
    if (getGraphExecutorOptimize()) {
      preoptimizeGraph(*optimized_graph);
    }
    return *optimized_graph;
  }

  void clear_execution_info() override {
    std::lock_guard<std::recursive_mutex> lock(compile_mutex);
    for (auto& graph : optimized_graphs_) {
      graph.reset();
    }
    for (auto& executor : executors_) {
      executor.reset();
    }
  }

  const c10::QualifiedName& qualname() const override {
    return name_;
  }

  const std::string& name() const override {
    return name_.name();
  }

  // if this isn't yet defined, run its method_creator function
  void ensure_defined() override;

  size_t num_inputs() const override {
    return graph()->inputs().size();
  }

  Function& setSchema(FunctionSchema schema) override {
    schema_ = make_unique<FunctionSchema>(std::move(schema));
    return *this;
  }

  const FunctionSchema& getSchema() const override;

  std::string pretty_print_schema() const override {
    AT_ASSERT(schema_);
    std::stringstream ss;
    ss << *schema_;
    return ss.str();
  }

  GraphExecutorState getDebugState() {
    return get_executor().getDebugState();
  }

  bool is_optimized() const {
    TORCH_WARN(
        "GraphFunction::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
    return true;
  }

  void check_single_output() override {
    TORCH_CHECK(
        graph()->outputs().size() == 1,
        "Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs");
  }

  GraphExecutor& get_executor() override {
    ensure_defined();
    std::lock_guard<std::recursive_mutex> lock(compile_mutex);
    auto& executor = executors_[currentSpecialization()];
    if (executor) {
      return executor;
    }
    check_single_output();
    executor = GraphExecutor(optimized_graph(), name_.name());
    return executor;
  }

 private:
  enum SpecializationKey {
    AutocastOff,
    AutocastOn,

    // This provides the number of specializations
    // (Must be last entry)
    TotalCount
  };

  SpecializationKey currentSpecialization() const;

 private:
  c10::QualifiedName name_;
  // The original, non-optimized graph
  std::shared_ptr<Graph> graph_; // for debugging and for inlining

  // Optimized graph, computed lazily. Used for inlining.
  mutable c10::optional<std::shared_ptr<Graph>>
      optimized_graphs_[SpecializationKey::TotalCount];

  // GraphFunctions are invokable from multiple threads, so this lock needs to
  // be held when we're initializing graph executor for the first time or
  // computing the optimized graph. We're using reentrant mutex so that we don't
  // need to worry about causing a deadlock by calling one method from another
  // (e.g. optimized_graph() from get_executor()).
  mutable std::recursive_mutex compile_mutex;

  // executor_[0] - autocast off
  // executor_[1] - autocast on
  GraphExecutor executors_[SpecializationKey::TotalCount];

  // an optional function that actually creates the method when
  // ensure_defined() is called. This is used by the compiler so
  // that it can construct methods out of order
  std::function<void(GraphFunction&)> function_creator_;

  // if absent, then we generate a default schema based on the graph
  // mutable because getSchema caches the default schema if one is requested
  // before a call to setSchema
  mutable std::unique_ptr<FunctionSchema> schema_;
};
} // namespace jit
} // namespace torch
