#pragma once
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

namespace torch {
namespace jit {

enum class FusionBehavior { STATIC, DYNAMIC };

using FusionStrategy = std::vector<std::pair<FusionBehavior, size_t>>;
// FusionStrategy is used to control the type and number of specializations that
//   can occur during fusion
//
// Usage: provide a list of pairs (type, depth) where type is one of "STATIC" or
//   "DYNAMIC" and depth is an integer.
//
// Behavior - static vs dynamic:
// - in STATIC fusion, fused ops are compiled to have fixed input shapes. The
//   input shapes are determined based on a number of initial profiling runs.
//   The shape is determined based on some initial profiling runs. For example,
//   if on the first run an input of shape [2, 4] is observed, then the compiled
//   op will only work on shapes of size [2, 4].
// - in DYNAMIC fusion, fused ops are compiled to have variable input shapes, so
//   that multiple shapes are possible. Dynamic fusion uses "symbolic shapes",
//   where any dimensions of the same value that are observed in profiling runs
//   are assumed to have the same value. For example, if inputs of [2,3,4] and
//   [3,4,5] are observed, then it is assumed that future inputs will have
//   shapes [a,b,c] and [b,c,d] for some values of a,b,c,d.
//
//   In both cases, we also recompile on new striding behavior, device, or dtype.
//
// Behavior - fallback functions & depth:
//   When an input doesn't match the format required by the specialized compiled
//   op, it will run a fallback function.
//   Fallback functions can also recursively be compiled and specialized based
//   on the input shape. Since compilation can be slow, the "depth" parameter is
//   provided to limit the number of specializations that can be compiled,
//   before JIT gives up on recompiling and falls back to a completely un-fused,
//   un-specialized implementation.
//
// The list of (type, depth) pairs controls the type of specializations and the
//   number of specializations. For example: [("STATIC", 2), ("DYNAMIC", 2)]
//   indicates that the first two specializations will use static fusions, the
//   following two specializations will use dynamic fusion, and any inputs that
//   satisfy none of the 4 options will run an unfused implementation.
// Below an example of the fallback function structure is shown, if given a
//   strategy of [("STATIC", 2), ("DYNAMIC", 2)] and if consecutive runs had
//   these input shapes:
//     [2, 2], [3, 3], [4, 4], [3, 5], ...
//
//   + specialized: statically fused, shape [2, 2]
//    \-> + fallback 1; statically fused, shape [3, 3]
//         \-> + fallback 2; dynamically fused, shape [A, A]
//              \-> + fallback 3: dynamically fused, shape [A, B]
//                   \-> final fallback: unspecialized, unfused
TORCH_API FusionStrategy getFusionStrategy();
// returns previous strategy
TORCH_API FusionStrategy setFusionStrategy(FusionStrategy& fusion_strategy);


struct TORCH_API ProfilingGraphExecutorImpl : public GraphExecutorImplBase {
  ProfilingGraphExecutorImpl(
      const std::shared_ptr<Graph>& graph,
      std::string function_name);

  const ExecutionPlan& getPlanFor(Stack& stack, size_t remaining_bailout_depth)
      override;
  GraphExecutorState getDebugState() override;
  ~ProfilingGraphExecutorImpl() override = default;

  void debugFlushCompilationCache() {
    std::lock_guard<std::mutex> lock(compile_mutex);
    pr_.reset();
    fallback_plan_.reset();
    profiling_plan_.reset();
    optimized_plan_.reset();
    // prevent memory leaks
    fallback_functions_.clear();
  remaining_bailout_depth_.reset();
  }

  bool isOptimized() const override {
    return optimized_plan_.has_value();
  }

 private:
  const ExecutionPlan& getOptimizedPlanFor(
      Stack& stack,
      size_t remaining_bailout_depth);
  void runProfilingInsensitiveOptimizations(std::shared_ptr<Graph>& graph);
  void runProfilingOptimizations(std::shared_ptr<Graph>& graph, size_t remaining_depth);
  void replaceFallbackGraphWithFallbackFunction(Block* b);
  std::unique_ptr<ProfilingRecord> pr_;
  c10::optional<ExecutionPlan>
      profiling_plan_; // plan to run in order to profiling the code
  c10::optional<ExecutionPlan> optimized_plan_;
  // this plan is used if getGraphExecutorOptimize is unset
  c10::optional<ExecutionPlan> fallback_plan_;
  // fallback functions are inserted for tensorexpr fusion groups
  // and by specialize_autogradzero. Whenever, at runtime, input
  // tensor don't match profiled properties, fallback functions are called
  // They are the deoptimized version of the logic in fusion groups
  // and/or autograd.
  // The fallback functions are owned by a GraphExecutor instance
  // They only exist in the optimized graph which is a private property
  // of the GraphExecutor and only shared with InterpreterState
  std::vector<std::unique_ptr<Function>> fallback_functions_;
  c10::optional<size_t> remaining_bailout_depth_;
};

} // namespace jit
} // namespace torch
