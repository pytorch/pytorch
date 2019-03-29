#include <torch/csrc/jit/passes/quantization.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

#include <stack>

namespace torch {
namespace jit {

void ExpandFakeQuantNodes(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void PropagateQuantInfo(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

static void addObserverFor(Value* v, Node* original_observer_node) {
  Node* def = v->node();
  WithInsertPoint ins(def);

  // We need to pass the value name to observer function - create a constant
  // holding this name.
  Value* vname = def->owningGraph()->insertConstant(v->uniqueName());

  // Create a new observer node. We just need to clone the original one.
  Node* observerNode =
      def->owningGraph()
          ->createClone(
              &*original_observer_node, [&](Value* v) { return v; }, false)
          ->insertAfter(def);

  // Set the type and the name of the output of the new observer node. It will
  // be used instead of the original value v.
  Value* observedValue = observerNode->addOutput();
  observedValue->setType(v->type());
  observedValue->setUniqueName(v->uniqueName() + ".observed");

  // Replace the uses of v with observedValue. We need to do it *before* we add
  // the inputs - otherwise we would replace the newly added inputs as well.
  v->replaceAllUsesWith(observedValue);

  // Now we can add the inputs.
  observerNode->addInput(v);
  observerNode->addInput(vname);
}

static bool outputsNeedToBeObserved(Node* n) {
  return n->kind().toQualString() != std::string("prim::Constant");
}

void InsertObserverNodes(std::shared_ptr<Graph>& graph, Node* observer_node) {
  // For storing all values that need to be instrumented with an observer call.
  std::vector<Value*> values_to_observe;

  // For traversing all blocks in the graph including subblocks.
  std::stack<Block*> blocks_to_visit;

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // Skip nodes that we don't need to observe, e.g. 'prim::Constant'.
      if (!outputsNeedToBeObserved(n)) {
        continue;
      }

      // Record all outputs in the values_to_observe - we'll later add observers
      // for all values from it.
      for (Value* v : n->outputs()) {
        values_to_observe.push_back(v);
      }

      // Schedule subblocks (if any) for visiting.
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // Actually add observer nodes.
  for (Value* v : values_to_observe) {
    addObserverFor(v, observer_node);
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
