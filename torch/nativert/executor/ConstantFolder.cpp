#include <torch/nativert/executor/ConstantFolder.h>

#include <algorithm>
#include <queue>

#include <c10/util/Enumerate.h>

#include <torch/nativert/executor/Weights.h>

namespace torch::nativert {

/*
  side effects:
    1. nodes deemed const-foldable nodes are unlinked from the graph.
       they are still owned by the graph (i.e., show up in graph.nodeOwner_)
       but are not accessible through the node iterator.

    2. kernels associated with const-foldable nodes are removed from the
       'kernels' input

    3. mark values deemed foldable as such, removing their producers
*/

void ConstantFolder::unlinkConstants(
    std::vector<std::unique_ptr<OpKernel>>& kernels) {
  TORCH_CHECK(
      kernels.size() == graph_.nodes().size(),
      "graph node count and kernel count should be equal");

  unlinked_ = true;

  /* resolve all of the nodes that are const foldable */

  c10::FastMap<Node*, uint32_t> nodeDynInputs;
  nodeDynInputs.reserve(graph_.nodes().size());

  c10::FastMap<const Node*, std::unique_ptr<OpKernel>*> nodeKernels;
  nodeKernels.reserve(graph_.nodes().size());

  const auto* input = &*graph_.nodes().begin();
  const auto* output = &*graph_.nodes().end();

  c10::FastSet<const Node*> run_const_graph_nodes;

  { // ignore prim.Input and prim.Output
    auto ct = 0;
    for (auto& n : graph_.nodes()) {
      if (&n == input || &n == output) {
        continue;
      }
      nodeDynInputs[&n] = n.numInputs();
      nodeKernels[&n] = &kernels[++ct];

      if (n.target() == "torch.ops.higher_order.run_const_graph") {
        run_const_graph_nodes.insert(&n);
      }
    }
  }

  for (const auto* run_const_graph_node : run_const_graph_nodes) {
    for (auto* user : run_const_graph_node->users()) {
      if (user == input || user == output) {
        continue;
      }
      nodeDynInputs[user] -= 1;
    }
  }

  const auto& inputsToWeights = graph_.signature().inputsToWeights();
  for (const auto& [inputName, weightName] : inputsToWeights) {
    for (auto* user : graph_.getValue(inputName)->users()) {
      if (user == input || user == output) {
        continue;
      }
      nodeDynInputs[user] -= 1;
    }
  }

  // set of foldable nodes for dedupe purposes
  c10::FastSet<const Node*> foldable;

  std::queue<Node*> constFoldableCandidates;
  for (auto& [node, ct] : nodeDynInputs) {
    if (ct++ /* will be decremented once dequeued */ == 0) {
      constFoldableCandidates.push(node);
    }
  }

  while (!constFoldableCandidates.empty()) {
    auto* candidate = constFoldableCandidates.front();
    constFoldableCandidates.pop();
    if (auto& ct = nodeDynInputs[candidate]; --ct == 0) {
      foldable.insert(candidate);
      Foldable f;
      f.node = candidate;
      f.kernel = std::move(*nodeKernels[candidate]);
      foldables_.push_back(std::move(f));

      candidate->unlink();

      for (auto* user : candidate->users()) {
        if (user == output) {
          continue;
        }
        if (foldable.find(user) == foldable.end()) {
          constFoldableCandidates.push(user);
        }
      }

      for (auto* out : candidate->outputs()) {
        auto* value = graph_.getValue(out->name());

        value->setIsFolded();

        // we only store folded values if there is a non-foldable user
        if (const auto& users = value->users();
            std::any_of(users.begin(), users.end(), [&](const auto* u) {
              return foldable.find(u) == foldable.end();
            })) {
          foldedOutputValueIds_.insert(value->id());
        }
      }
    }
  }

  for (const auto& f : foldables_) {
    VLOG(1) << "Const-folded node: " << *f.node;
  }
  LOG(INFO) << "Const-folded " << foldables_.size() << " nodes";

  // remove moved (i.e., associated w/ const-folded nodes) kernels
  // from the input kernel vector
  kernels.erase(
      std::remove_if(
          kernels.begin(),
          kernels.end(),
          [](const auto& k) { return k == nullptr; }),
      kernels.end());

  graph_.renumberValues();
  graph_.finalize();
  graph_.lint();

  return;
}

/*
  side effects:
    1. weights whose users are ONLY const-foldable nodes will be removed
       from the 'weights' input
*/

void ConstantFolder::evaluate(Weights& weights) {
  TORCH_CHECK(
      unlinked_,
      "cannot evaluate weights for a graph whose constants have not been unlinked via ConstFolder::unlinkConstants");

  weights.validateAllWeightsLoaded();

  ExecutionFrame frame(graph_);
  frame.setWeights(weights);

  c10::FastMap<std::string, c10::IValue> foldedValues;

  for (const auto& f : foldables_) {
    f.kernel->compute(frame);

    for (auto&& [i, out] : c10::enumerate(f.node->outputs())) {
      if (foldedOutputValueIds_.find(out->id()) !=
          foldedOutputValueIds_.end()) {
        foldedValues[std::string{out->name()}] = f.kernel->output(i, frame);
      }
    }
  }

  for (auto it = std::make_move_iterator(foldedValues.begin());
       it != std::make_move_iterator(foldedValues.end());
       ++it) {
    auto [n, iv] = std::move(*it);
    weights.setConstFoldedValue(n, std::move(iv));
  }
}

} // namespace torch::nativert
