#pragma once

#include "torch/csrc/nativert/graph/Graph.h"

#include <torch/script.h>

namespace torch::nativert {

class Graph;

// ExecutionPlan is the result produced by ExecutionPlanner
// ATM, it only contains value deallocation plan.
// In the future, it can include execution order plan, allocation plan for
// parameter/gradient alignment, static memory plan for activation buffer reuse
// ect...
struct ExecutionPlan {
  // i-th entry in this list are the Values can be freed *after* execution i-th
  // node
  std::vector<std::vector<ValueId>> valuesToFree;
};

class ExecutionPlanner {
 public:
  explicit ExecutionPlanner(const Graph& graph) : graph_(graph) {}

  std::unique_ptr<ExecutionPlan> createPlan();
  // get list of values we can't free
  static c10::FastSet<ValueId> staticValues(const Graph& graph);

 private:
  void generateDeallocationPlan(ExecutionPlan& plan);

  // NYI
  void generatedMemoryPlan(ExecutionPlan& plan);

  const Graph& graph_;
};

std::ostream& operator<<(std::ostream& out, const ExecutionPlan& plan);

} // namespace torch::nativert
