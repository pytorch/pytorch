#include "torch/csrc/jit/passes/annotate_effects.h"
#include <set>

namespace torch {
namespace jit {
namespace {

// Ops with effects that we need to track through the passing of world tokens
// e.g. mutation
const std::unordered_set<Symbol> effectfulOps = {
    aten::append,
};

/**
 * AnnotateEffects
 *
 * This pass annotates effectful operations (such as ones that mutate existing
 * values) to prevent subsequent passes from re-ordering ops in a way that
 * changes the meaning of the program.
 *
 * It does this by adding primitive memory fence ops around nodes that use
 * mutable values. The memory fences input and output a "World" token, which
 * expresses effects explicitly in the IR and forces all fenced nodes to be
 * linearized.
 */
class AnnotateEffectsImpl {
 public:
  void annotateEffects(Graph* g) {
    // Generate the first world token
    const auto tokenGenerator = g->create(prim::Constant);
    g->block()->prependNode(tokenGenerator);
    auto curToken = tokenGenerator->output()->setType(WorldType::get());

    visitBlock(g->block(), curToken);
  }

 private:
  Value* visitBlock(Block* block, Value* curToken) {
    for (auto* node : block->nodes()) {
      curToken = visitNode(node, curToken);
    }
    return curToken;
  }

  // If a node uses a mutable variable (or mutates a previously constant
  // variable), create a memory fence around the node.
  //
  // Returns the last world token emitted for subsequent memory fences to use.
  Value* visitNode(Node* node, Value* curToken) {
    if (node->kind() == prim::If) {
      JIT_ASSERT(node->blocks().size() == 2);

      auto trueBlock = node->blocks().at(0);
      auto falseBlock = node->blocks().at(1);

      auto trueToken = visitBlock(trueBlock, curToken);
      auto falseToken = visitBlock(falseBlock, curToken);

      // If any branch has a mutating op, this node has to output a world token
      if (trueToken != curToken || falseToken != curToken) {
        trueBlock->registerOutput(trueToken);
        falseBlock->registerOutput(falseToken);

        return node->addOutput()->setType(WorldType::get());
      }
      return curToken;
    }

    if (node->kind() == prim::Loop) {
      JIT_ASSERT(node->blocks().size() == 1);
      auto block = node->blocks().at(0);
      auto newToken = visitBlock(block, curToken);

      if (newToken != curToken) {
        // Register the world token as a loop-carried dependency
        block->addInput()->setType(WorldType::get());
        block->registerOutput(curToken);

        // Thread the world token through the loop node.
        node->addInput(curToken);
        return node->addOutput()->setType(WorldType::get());
      }
      return curToken;
    }

    if (shouldAddFence(node)) {
      return addFenceForNode(node, curToken);
    }

    return curToken;
  }

  bool shouldAddFence(Node* node) {
    // Check if this node uses a known mutable value
    for (auto* input : node->inputs()) {
      if (mutableValues_.count(input) != 0) {
        return true;
      }
    }

    // Otherwise, check if this node is the first mutation of a value
    if (effectfulOps.count(node->kind()) != 0) {
      const auto inputs = node->inputs();
      Value* mut = inputs.at(0);
      JIT_ASSERT(mut->type()->kind() == TypeKind::ListType); // TODO

      mutableValues_.insert(mut);
      return true;
    }

    return false;
  }

  // Create a memory fence around a node, using the world token
  // Input:
  //   = aten::append(%list, %new_element)
  //
  // Output:
  //  %t.1 : World, %list.2 : int[] = prim::MemoryFence(%curToken, %list)
  //   = aten::append(%list.2, %new_element)
  //  %t.2 : World, %list.3 : int[] = prim::MemoryFence(%t.1, %list2)
  //
  // Returns the new world token (%t.2) for subsequent fences to use.
  Value* addFenceForNode(Node* node, Value* curToken) {
    auto mut = node->inputs().at(0);

    // Add a start fence
    auto startFence =
        node->owningGraph()->create(prim::MemoryFence, /*outputs=*/2);
    startFence->addInput(curToken);
    startFence->addInput(mut);
    curToken = startFence->outputs()[0]->setType(WorldType::get());
    mut = startFence->outputs()[1]->setType(ListType::ofInts());
    startFence->insertBefore(node);

    // modify the node to take in the start fence's output list
    node->replaceInput(0, mut);

    // Add an end fence
    auto endFence =
        node->owningGraph()->create(prim::MemoryFence, /*outputs=*/2);
    endFence->addInput(curToken);
    endFence->addInput(mut);
    curToken = endFence->outputs()[0]->setType(WorldType::get());
    mut = endFence->outputs()[1]->setType(ListType::ofInts());
    endFence->insertAfter(node);

    return curToken;
  }

  std::set<Value*> mutableValues_;
};

} // namespace

void AnnotateEffects(std::shared_ptr<Graph>& graph) {
  AnnotateEffectsImpl impl;
  impl.annotateEffects(graph.get());
}

} // namespace jit
} // namespace torch
