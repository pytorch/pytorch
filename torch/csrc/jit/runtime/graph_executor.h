#pragma once

#include <atomic>
#include <memory>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/variable_tensor_list.h>

TORCH_DECLARE_bool(torch_jit_enable_new_executor);

TORCH_DECLARE_bool(torch_jit_execution_plan_reuse_code_graph);

namespace torch::jit {
struct GraphExecutorState;
struct Code;

enum ExecutorExecutionMode {
  SIMPLE,
  PROFILING,
};

struct ExecutionPlan {
  ExecutionPlan() = default;
  ExecutionPlan(std::shared_ptr<Graph> graph, std::string function_name)
      : code(graph, std::move(function_name)),
        graph(
            FLAGS_torch_jit_execution_plan_reuse_code_graph
                ? code.graph()
                : std::move(graph)) {}

  operator bool() const {
    return static_cast<bool>(graph);
  }

  Code code;
  std::shared_ptr<Graph> graph;
};

// Notice that those structs don't manage lifetime of their members.
// They are only valid only right after you call getDebugState() and should
// never be used again once another GraphExecutor function is called.

struct GraphExecutorState {
  const Graph* graph = nullptr;
  ExecutionPlan fallback; // XXX: members of this field are optional
  std::unordered_map<ArgumentSpec, ExecutionPlan> execution_plans;
};

struct TORCH_API EnableProfilingGuard {
  EnableProfilingGuard();
  ~EnableProfilingGuard();

 private:
  bool old_executor_mode = false;
  bool old_get_optimize = false;
};

struct GraphExecutorImplBase;
struct TORCH_API GraphExecutor {
  GraphExecutor() = default;
  GraphExecutor(const std::shared_ptr<Graph>& graph, std::string function_name);

  GraphExecutor(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      ExecutorExecutionMode executor_mode);

  void run(Stack& inputs);
  c10::intrusive_ptr<Future> runAsync(
      Stack& stack,
      TaskLauncher taskLauncher = at::launch);

  // `remaining_bailout_depth` stands for the maximum number of profiled and
  // specialized recompilations allowed for the current `GraphExecutor`. if
  // remaining_bailout_depth is equal to 0, `GraphExecutor` won't perform any
  // profiling and specialization. This is also equivalent to the
  // SIMPLE_EXECUTOR mode. if remaining_bailout_depth is greater than 0,
  // `GraphExecutor` will profile and specialize its input graph based on the
  // profiled information whenever a bailout check is failed/triggered, a new
  // `GraphExecutor` will be created. This new `GraphExecutor`'s
  // remaining_bailout_depth will be reduced by 1.
  // If no bailout depth is passed, the depth will be initialized from the
  // current global fusion strategy settings.
  const ExecutionPlan& getPlanFor(
      Stack& inputs,
      std::optional<size_t> remaining_bailout_depth = std::nullopt);
  GraphExecutorState getDebugState();

  void debugFlushCompilationCache();

  bool isOptimized() const;

 private:
  std::shared_ptr<GraphExecutorImplBase> pImpl;
};

TORCH_API Node* replaceBlockWithFallbackGraph(
    Block* b,
    ArrayRef<Value*> inputs);

// These passes need to run before it is valid to pass to the interpreter
// regardless of whether sizes have been specialized or not.
TORCH_API void runRequiredPasses(const std::shared_ptr<Graph>& g);

TORCH_API void debugSetFusionGroupInlining(bool state);
TORCH_API bool getFusionGroupInlining();

TORCH_API void debugSetAutodiffSubgraphInlining(bool state);
TORCH_API std::shared_ptr<Graph> lastExecutedOptimizedGraph();

TORCH_API std::atomic<bool>& getProfilingMode();
TORCH_API std::atomic<bool>& getExecutorMode();
TORCH_API std::atomic<size_t>& getNumProfiledRuns();
TORCH_API size_t getBailoutDepth();
TORCH_API bool IsNewExecutorEnabled();

struct TORCH_API GraphOptimizerEnabledGuard {
  GraphOptimizerEnabledGuard(bool state)
      : old_state_(getGraphExecutorOptimize()) {
    setGraphExecutorOptimize(state);
  }

  ~GraphOptimizerEnabledGuard() {
    setGraphExecutorOptimize(old_state_);
  }

  bool old_state_;
};

namespace detail {

GraphExecutor* getGradExecutor(Operation& op);

GraphExecutor* getDifferentiableGraphOpExecutor(Operation& op);

// for debugging information we expose a way to get the last actually
// run graph. Previous approaches allowed querying the GraphExecutor
// for what graph it would run in certain circumstances (graphFor), but
// this is fragile because we sometimes change how these decisions are made.
// This interface still allows our tests to look at optimized graphs, but
// with less plumbing.
} // namespace detail

} // namespace torch::jit
