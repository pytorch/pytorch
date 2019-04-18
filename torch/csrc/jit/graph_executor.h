#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/autodiff.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/variable_tensor_list.h>
#include <memory>

namespace torch {
namespace jit {

struct GraphExecutorState;

// Notice that those structs don't manage lifetime of their members.
// They is only valid only right after you call getDebugState() and should never
// be used again once another GraphExecutor function is called.
struct ExecutionPlanState {
  Code* code = nullptr;
  const Graph* graph = nullptr;
};

struct GraphExecutorState {
  const Graph* graph = nullptr;
  ExecutionPlanState fallback; // XXX: members of this field are optional
  std::unordered_map<ArgumentSpec, ExecutionPlanState> execution_plans;
};

struct GraphExecutorImpl;
struct TORCH_API GraphExecutor {
  GraphExecutor() = default;
  GraphExecutor(std::shared_ptr<Graph> graph, bool optimize = true);
  void run(Stack& inputs);
  explicit operator bool() const {
    return pImpl != nullptr;
  }
  std::shared_ptr<Graph> graph() const;
  GraphExecutorState getDebugState();

 private:
  std::shared_ptr<GraphExecutorImpl> pImpl;
};

// These passes need to run before it is valid to pass to the interpreter
// regardless of whether sizes have been specialized or not.
TORCH_API void runRequiredPasses(const std::shared_ptr<Graph>& g);

TORCH_API void debugSetAutodiffSubgraphInlining(bool state);
TORCH_API std::shared_ptr<Graph> lastExecutedOptimizedGraph();

namespace detail {

GraphExecutor* getGradExecutor(Operation& op);

// for debugging information we expose a way to get the last actually
// run graph. Previous approaches allowed querying the GraphExecutor
// for what graph it would run in certain circumstances (graphFor), but
// this is fragile because we sometimes change how these decisions are made.
// This interface still allows our tests to look at optimized graphs, but
// with less plumbing.
} // namespace detail

} // namespace jit
} // namespace torch
