#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch {
namespace jit {

struct ExecutionPlan {
  ExecutionPlan() = default;
  ExecutionPlan(
      std::shared_ptr<Graph> graph,
      std::string function_name,
      size_t remaining_bailout_depth = 0)
      : code(graph, std::move(function_name), remaining_bailout_depth),
        graph(std::move(graph)) {}

  operator bool() const {
    return static_cast<bool>(graph);
  }

  Code code;
  std::shared_ptr<Graph> graph;
};

// Notice that those structs don't manage lifetime of their members.
// They is only valid only right after you call getDebugState() and should never
// be used again once another GraphExecutor function is called.

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct GraphExecutorState {
  const Graph* graph = nullptr;
  ExecutionPlan fallback; // XXX: members of this field are optional
  std::unordered_map<ArgumentSpec, ExecutionPlan> execution_plans;
};

class TORCH_API Executor {
 public:
  virtual const ExecutionPlan& getPlanFor(
      Stack& inputs,
      size_t remaining_bailout_depth) = 0;
  virtual GraphExecutorState getDebugState() = 0;
  virtual void debugFlushCompilationCache() = 0;

  virtual ~Executor() = default;
};

} // namespace jit
} // namespace torch
