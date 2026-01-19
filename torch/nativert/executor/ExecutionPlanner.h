#pragma once

#include <c10/util/FbcodeMaps.h>

#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

// ExecutionPlan is the result produced by ExecutionPlanner
// ATM, it only contains value deallocation plan.
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
  const Graph& graph_;
};

std::ostream& operator<<(std::ostream& out, const ExecutionPlan& plan);

} // namespace torch::nativert
