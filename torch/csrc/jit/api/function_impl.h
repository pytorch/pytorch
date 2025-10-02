#pragma once

#include <ATen/core/function.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch::jit {

struct TORCH_API GraphFunction : public Function {
  GraphFunction(
      c10::QualifiedName name,
      std::shared_ptr<Graph> graph,
      std::function<void(GraphFunction&)> function_creator,
      std::optional<ExecutorExecutionMode> executor_execution_mode =
          std::nullopt)
      : name_(std::move(name)),
        graph_(std::move(graph)),
        executor_execution_mode_(executor_execution_mode),
        function_creator_(std::move(function_creator)) {}

  bool isGraphFunction() const override {
    return true;
  }

  void run(Stack& stack) override;

  std::function<void(GraphFunction&)> function_creator() const {
    return function_creator_;
  }

  c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch) override;

  std::shared_ptr<Graph> graph() const {
    return graph_;
  }

  std::shared_ptr<Graph> optimized_graph() const;

  const c10::QualifiedName& qualname() const override {
    return name_;
  }

  // private/unstable api. sets the initial execution mode
  // will not affect executor if there is an existing executor
  // created for this function
  void _set_initial_executor_execution_mode(ExecutorExecutionMode mode) {
    executor_execution_mode_ = mode;
  }
  // private/unstable api. sets flag of whether or not to ignore amp.
  // will not affect executor if there is an existing executor
  // created for this function
  void _set_ignore_amp(bool ignore_amp) {
    force_no_amp_ = ignore_amp;
  }

  // if this isn't yet defined, run its method_creator function
  void ensure_defined() override;

  size_t num_inputs() const override {
    return graph()->inputs().size();
  }

  Function& setSchema(FunctionSchema schema) override {
    schema_ = std::make_unique<FunctionSchema>(std::move(schema));
    return *this;
  }

  const FunctionSchema& getSchema() const override;

  GraphExecutorState getDebugState() {
    return get_executor().getDebugState();
  }

  bool is_optimized() const {
    TORCH_WARN(
        "GraphFunction::is_optimized() is deprecated and always returns true. "
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
    std::lock_guard<std::recursive_mutex> lock(compile_mutex);
    auto& executor = executors_[currentSpecialization()];
    if (executor) {
      return *executor;
    }
    check_single_output();
    const std::string& name = name_.name();
    std::shared_ptr<Graph> opt_graph = optimized_graph();
    if (!executor_execution_mode_) {
      executor = GraphExecutor(opt_graph, name);
    } else {
      executor = GraphExecutor(opt_graph, name, *executor_execution_mode_);
    }
    return *executor;
  }

  using Function::call;
  bool call(
      Stack& stack,
      std::optional<size_t> bailOut,
      c10::function_ref<void(const Code&)> f) override {
    f(get_executor().getPlanFor(stack, bailOut).code);
    return true;
  }

  void clear_optimized_graphs() {
    optimized_graphs_.fill(nullptr);
  }

 private:
  enum SpecializationKey {
    AutocastOff,
    CpuAutocastOn,
    GpuAutocastOn,
    CpuGpuAutocastOn,

    // This provides the number of specializations
    // (Must be last entry)
    TotalCount
  };

  SpecializationKey currentSpecialization() const;

 private:
  c10::QualifiedName name_;
  // The original, non-optimized graph
  std::shared_ptr<Graph> graph_; // for debugging and for inlining

  // allows users to specify Simple/Profiling Executor for function
  // TODO: add more executors
  mutable std::optional<ExecutorExecutionMode> executor_execution_mode_;

  // if invoked on a graph that has already traced through amp
  // don't invoke amp pass
  mutable bool force_no_amp_ = false;
  // Optimized graph, computed lazily. Used for inlining.
  mutable std::array<std::shared_ptr<Graph>, SpecializationKey::TotalCount>
      optimized_graphs_;

  // GraphFunctions are invocable from multiple threads, so this lock needs to
  // be held when we're initializing graph executor for the first time or
  // computing the optimized graph. We're using reentrant mutex so that we don't
  // need to worry about causing a deadlock by calling one method from another
  // (e.g. optimized_graph() from get_executor()).
  mutable std::recursive_mutex compile_mutex;

  // executor_[0] - autocast off
  // executor_[1] - autocast cpu on
  // executor_[2] - autocast gpu on
  // executor_[3] - autocast cpu & gpu on
  std::array<std::optional<GraphExecutor>, SpecializationKey::TotalCount>
      executors_;

  // an optional function that actually creates the method when
  // ensure_defined() is called. This is used by the compiler so
  // that it can construct methods out of order
  std::function<void(GraphFunction&)> function_creator_;

  // if absent, then we generate a default schema based on the graph
  // mutable because getSchema caches the default schema if one is requested
  // before a call to setSchema
  mutable std::unique_ptr<FunctionSchema> schema_;
};

// Short hands for dynamic_cast<GraphFunction*>.
TORCH_API GraphFunction* tryToGraphFunction(Function&) noexcept;
TORCH_API GraphFunction& toGraphFunction(Function&);
TORCH_API const GraphFunction& toGraphFunction(const Function&);
} // namespace torch::jit
C10_DECLARE_bool(torch_jit_do_not_store_optimized_graph);
