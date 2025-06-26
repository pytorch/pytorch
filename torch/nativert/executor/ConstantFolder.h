#pragma once

#include <memory>
#include <vector>

#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert {

struct Foldable {
  Node* node;
  std::unique_ptr<OpKernel> kernel;
};

class ConstantFolder {
 public:
  explicit ConstantFolder(Graph& graph) : graph_(graph) {}

  /*
    1. identify nodes without dynamic inputs, mark as foldable

    2. traverse the nodes deemed foldable as if they were being evaluated,
       pushing nodes that become foldable after it's inputs were traversed.

       unlink foldable nodes from the graph in the topological order in which
       they were traversed, storing the node and its associated kernel (moved
       from 'kernels') as a foldable in Constantfolder
  */
  void unlinkConstants(
      /* kernels for const-foldable nodes will be removed from this vector */
      std::vector<std::unique_ptr<OpKernel>>& kernels);

  /*
    1. execute foldables_ on an execution frame initialized with the passed-in
    weights, calling Weights::setConstFoldedValue if the folded value is
    consumed by a non-foldable node
  */
  void evaluate(Weights& weights);

 private:
  Graph& graph_;
  // unlinked nodes sorted in their topological order
  // s.t., they can be evaluated sequentially
  std::vector<Foldable> foldables_;

  bool unlinked_{false};

  c10::FastSet<ValueId> foldedOutputValueIds_;
};

} // namespace torch::nativert
