#include "torch/csrc/jit/passes/annotate_effects.h"

#include <set>
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {
namespace {

/**
 * AnnotateEffects
 *
 * This pass annotates effectful operations (such as ones that mutate existing
 * values) to prevent subsequent passes from re-ordering ops in a way that
 * changes the meaning of the program.
 *
 * It does this by threading a "world token" value through nodes that use
 * mutable values. This models effects explicitly in the IR and forces all
 * annotated nodes to be linearized during optimization.
 *
 * The world token is threaded directly through any nodes that mutate values.
 * For purely functional operators, their node will be "fenced" by two
 * `prim::MemoryFence` nodes that take world tokens as their input.
 */
class AnnotateEffectsImpl {
 public:
  void annotateEffects(Graph* g) {
    // TODO(suo): We need to change this so that the world token is an input
    // and output of the graph. That would require changing a bunch of interop
    // and batching code, so leaving it out for now.
    //
    // auto curToken = g->addInput("world")->setType(WorldType::get());
    // curToken = visitBlock(g->block(), curToken);
    // g->registerOutput(curToken);

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
  // variable), annotate it
  //
  // Returns the last world token emitted for subsequent annotations to use.
  Value* visitNode(Node* node, Value* curToken) {
    if (node->kind() == prim::MemoryFence) {
      return curToken;
    }

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
      if (!shouldAnnotate(block)) {
        // Bail out early if there's no mutable variables used inside
        return curToken;
      }

      // Register the world token as a loop carried dependency
      auto beginLoopToken = block->addInput()->setType(WorldType::get());
      auto endLoopToken = visitBlock(block, beginLoopToken);
      block->registerOutput(endLoopToken);

      JIT_ASSERT(endLoopToken != beginLoopToken);

      // Thread the world token through the loop node
      node->addInput(curToken);
      return node->addOutput()->setType(WorldType::get());
    }

    // For mutating ops, just thread the world token through the node.
    if (isMutatingOp(node)) {
      node->replaceInput(0, curToken);
      return node->outputs().at(0);
    }

    JIT_ASSERT(node->blocks().size() == 0);

    // For pure ops that need to be annotated, fence them.
    if (shouldAnnotate(node)) {
      return addFenceForNode(node, curToken);
    }

    return curToken;
  }

  bool shouldAnnotate(Node* node) {
    // Check if this node uses a known mutable value
    for (auto* input : node->inputs()) {
      if (input->type()->kind() != TypeKind::ListType) {
        // TODO(suo): Right now, we only support mutable lists.
        // If we remove this check, it's not clear whether:
        //
        //   append(int[] a, int b)
        //
        // mutates `a` or `b`. We'll need to extend the schema language to be
        // able to express which argument is mutated.
        continue;
      }
      // First check the cache
      if (mutableValues_.count(input) != 0) {
        return true;
      }

      // Check whether any mutating op uses this input
      for (const auto& use : input->uses()) {
        if (isMutatingOp(use.user)) {
          mutableValues_.insert(input);
          return true;
        }
      }
    }

    // Check that any sub-blocks need to be annotated
    for (auto block : node->blocks()) {
      if (shouldAnnotate(block)) {
        return true;
      }
    }

    return false;
  }

  bool shouldAnnotate(Block* block) {
    for (auto node : block->nodes()) {
      if (shouldAnnotate(node)) {
        return true;
      }
    }
    return false;
  }

  bool isMutatingOp(const Node* node) {
    return std::any_of(
        node->inputs().begin(), node->inputs().end(), [](const Value* input) {
          return input->type() == WorldType::get();
        });
  }

  // Create a memory fence around a node, using the world token
  //
  // Input:
  //  %size : Int = prim::len(%mut_list)
  //
  // Output:
  //  %t.1 : World, %list.2 : int[] = prim::MemoryFence(%curToken, %mut_list)
  //  %size : Int = prim::len(%mut_list)
  //  %t.2 : World, %size.2 : int = prim::MemoryFence(%t.1, %size)
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
    mut = startFence->outputs()[1]->setType(mut->type());

    startFence->insertBefore(node);

    // modify the node to take in the start fence's output list
    node->replaceInput(0, mut);

    // Add an end fence
    auto endFence =
        node->owningGraph()->create(prim::MemoryFence, /*outputs=*/2);

    endFence->addInput(curToken);
    endFence->addInput(node->outputs().at(0));

    curToken = endFence->outputs()[0]->setType(WorldType::get());
    mut = endFence->outputs()[1]->setType(node->outputs().at(0)->type());

    endFence->insertAfter(node);

    return curToken;
  }

  // Memoize which values will be mutated at some point in the program
  std::set<Value*> mutableValues_;
};
} // namespace

void AnnotateEffects(std::shared_ptr<Graph>& graph) {
  AnnotateEffectsImpl impl;
  impl.annotateEffects(graph.get());

  // Prune the dummy world tokens
  EliminateDeadCode(graph);
}

} // namespace jit
} // namespace torch
