#include <torch/csrc/jit/passes/quantization.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace {} // namespace

void ExpandFakeQuantNodes(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void PropagateQuantInfo(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

static void addObserverFor(torch::jit::Value* v) {
  torch::jit::Node* def = v->node();
  torch::jit::Node* observer =
      def->owningGraph()->create(at::Symbol::fromQualString("aten::observer"));
  def->owningBlock()->appendNode(observer);
  observer->moveAfter(def);
  v->replaceAllUsesWith(observer->outputs()[0]);
  observer->outputs()[0]->setUniqueName(v->uniqueName() + ".observed");
  observer->addInput(v);
}

void InsertObserverNodes(std::shared_ptr<Graph>& graph) {
  for (const auto& n : graph->nodes()) {
    // Skip nodes that we've just added
    if (n->kind().toQualString() == std::string("aten::observer")) {
      continue;
    }

    // Add observers to all outputs of node N unless they are inside a subgraph
    // that we expect to be fused.
    for (torch::jit::Value* v : n->outputs()) {
      addObserverFor(v);
    }
  }
}

void InsertFakeQuantNodes(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void QuantLinting(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

} // namespace jit
} // namespace torch
