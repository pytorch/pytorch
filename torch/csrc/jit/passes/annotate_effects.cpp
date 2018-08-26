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
 * For mutating operators: the world token is threaded directly through the node
 * For purely functional operators: their node will be "fenced" by two
 *   `prim::MemoryFence` nodes that take world tokens as their input.
 *
 * Graphs have special EntryWorld and ExitWorld nodes that provide end-points
 * for the world token. They are similar to graph inputs/outputs in that they
 * are not in the node list and only accessible via special methods.
 *
 * When inlined, graphs will manifest the EntryWorld/ExitWorld nodes explicitly
 * so that they can act as endpoints where the callee "world thread" can be
 * joined to the caller world thread.
 */
class AnnotateEffectsImpl {
 public:
  void annotateEffects(Graph* g) {
    // Generate the first world token
    auto curToken = g->entryWorld()->addOutput()->setType(WorldType::get());
    auto lastToken = visitBlock(g->block(), curToken);

    g->exitWorld()->addInput(lastToken);
    g->exitWorld()->addOutput()->setType(WorldType::get());
  }

 private:
  Value* visitBlock(Block* block, Value* curToken) {
    for (auto* node : block->nodes()) {
      // Handle inlined functions. Inlined functions expose their Entry and
      // Exit tokens as regular nodes. These exposed nodes provide fixed points
      // to thread the current world token through.
      //
      // We can ignore all nodes in between the entry and exit tokens, since
      // tokens have already been threaded through then.
      bool skip = false;
      if (node->kind() == prim::EntryWorld) {
        curToken = visitEntryWorld(node, curToken);
        // Skip until we see the corresponding ExitWorld node.
        skip = true;
        continue;
      }

      if (node->kind() == prim::ExitWorld) {
        curToken = visitExitWorld(node, curToken);
        // Resume threading the token normally.
        skip = false;
        continue;
      }

      if (skip) {
        continue;
      }

      curToken = visitNode(node, curToken);
    }
    return curToken;
  }

  // Special handling for inlined functions.
  // Replace the inlined function's inital entry token with the outer function's
  // current token.
  Value* visitEntryWorld(Node* node, Value* curToken) {
    JIT_ASSERT(node->kind() == prim::EntryWorld);
    if (node->outputs().empty()) {
      return curToken;
    }
    auto inlinedEntryToken = node->output();
    inlinedEntryToken->replaceAllUsesWith(curToken);
    return curToken;
  }

  // Returns the inlined function's last world token directly
  Value* visitExitWorld(Node* node, Value* curToken) {
    JIT_ASSERT(node->kind() == prim::ExitWorld);
    if (node->outputs().empty()) {
      return curToken;
    }
    auto lastReturnedToken = node->input();
    auto inlinedExitToken = node->output();
    // There shouldn't be any uses of the exit token, so DCE should correctly
    // clean it up.
    JIT_ASSERT(inlinedExitToken->uses().empty());
    return lastReturnedToken;
  }

  // General node annotation. If a node uses a mutable variable (or mutates a
  // previously constant variable), annotate it
  //
  // Returns the last world token emitted for subsequent annotations to use.
  Value* visitNode(Node* node, Value* curToken) {
    // Avoid annotating memory fences. This avoids an infinite loop as we add
    // fences and continue to iterate through nodes.
    if (node->kind() == prim::MemoryFence) {
      // Return this memory fence's world token
      return node->outputs().at(0);
    }

    // Handle inlined functions. Inlined functions will expose their Entry and
    // Exit tokens as regular nodes. These exposed nodes provide fixed points
    // to thread the current world token through.
    if (node->kind() == prim::EntryWorld) {
      // Replace the inlined function's inital entry token with the outer
      // function's current token.
      auto inlinedEntryToken = node->output();
      inlinedEntryToken->replaceAllUsesWith(curToken);
      return curToken;
    }

    if (node->kind() == prim::ExitWorld) {
      auto lastReturnedToken = node->input();
      auto inlinedExitToken = node->output();
      // There shouldn't be any uses of the exit token, so DCE should correctly
      // clean it up.
      JIT_ASSERT(inlinedExitToken->uses().empty());
      return lastReturnedToken;
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
      // Replace the "dummy" token generated by the compiler
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

  bool shouldAnnotate(const Node* node) {
    // Check if this node uses a known mutable value
    for (const auto* input : node->inputs()) {
      if (!isMutableType(input)) {
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

  bool shouldAnnotate(const Block* block) {
    return std::any_of(
        block->nodes().begin(), block->nodes().end(), [this](const Node* node) {
          return shouldAnnotate(node);
        });
  }

  bool isMutableType(const Value* value) {
    return value->type()->kind() == TypeKind::ListType;
  }

  bool isMutatingOp(const Node* node) {
    return !node->inputs().empty() &&
        node->inputs()[0]->type() == WorldType::get();
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
    // Search for node's first mutable input.
    size_t mutableInputIdx = 0;
    for (const auto* input : node->inputs()) {
      if (mutableValues_.count(input) != 0) {
        break;
      }
      mutableInputIdx++;
    }

    auto mutableInput = node->inputs().at(mutableInputIdx);

    // Add a start fence
    auto startFence =
        node->owningGraph()->create(prim::MemoryFence, /*outputs=*/2);

    startFence->addInput(curToken);
    startFence->addInput(mutableInput);

    curToken = startFence->outputs()[0]->setType(WorldType::get());
    mutableInput = startFence->outputs()[1]->setType(mutableInput->type());

    startFence->insertBefore(node);

    // modify the node to take in the start fence's output value
    node->replaceInput(mutableInputIdx, mutableInput);

    // Add an end fence
    auto endFence =
        node->owningGraph()->create(prim::MemoryFence, /*outputs=*/2);

    endFence->addInput(curToken);
    endFence->addInput(node->outputs().at(0));

    curToken = endFence->outputs()[0]->setType(WorldType::get());
    mutableInput =
        endFence->outputs()[1]->setType(node->outputs().at(0)->type());

    endFence->insertAfter(node);

    return curToken;
  }

  // Memoize which values will be mutated at some point in the program
  std::set<const Value*> mutableValues_;
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
