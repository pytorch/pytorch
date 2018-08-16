#include "torch/csrc/jit/passes/annotate_effects.h"
#include <set>

namespace torch {
namespace jit {
namespace {

// Ops with effects that we need to track through the passing of world tokens
// e.g. mutation
const std::unordered_set<Symbol> effectfulOps = {
    prim::append,
    // TODO(suo): need to support the following side-effecting ops. This will
    // require that we know which value is being mutated. Probably the best
    // place to put that is in the function schema?
    //
    // prim::Print,
    // aten::permute,
    // aten::rand,
    // aten::rand_out,
    // aten::rand_like,
    // aten::randint,
    // aten::randint_out,
    // aten::randint_like,
    // aten::randn,
    // aten::randn_out,
    // aten::randn_like,
    // aten::randperm,
    // aten::randperm_out,
};

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
 * UnAnnotateEffects just does the opposite of annotate effects, removing all
 * world tokens so that there's no effect on the runtime.
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
      return annotateNode(node, curToken);
    }

    JIT_ASSERT(node->blocks().size() == 0);

    if (shouldAnnotate(node)) {
      return annotateNode(node, curToken);
    }

    return curToken;
  }

  bool shouldAnnotate(Node* node) {
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

    // Check that any sub-blocks
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

  // Annotate a node by threading the world token through it.
  // Input:
  //   = prim::append(%list, %new_element)
  //
  // Output:
  //  t.2 : World = prim::append(%list, %new_element, %t.1)
  //
  // Returns the new world token (%t.2).
  Value* annotateNode(Node* node, Value* curToken) {
    node->addInput(curToken);
    return node->addOutput()->setType(WorldType::get());
  }

  std::set<Value*> mutableValues_;
};

void eraseTokens(Node* node);
void eraseTokens(Block* block) {
  for (Node* node : block->nodes()) {
    eraseTokens(node);
  }
}
void eraseTokens(Node* node) {
  // Erase all uses of the world token this node outputs.
  size_t i = 0;
  for (auto output : node->outputs()) {
    if (output->type() == WorldType::get()) {
      for (const auto& use : output->uses()) {
        eraseTokens(use.user);
      }
      output->removeAllUses();
      break;
    }
    ++i;
  }

  // If this node outputs a world token, erase the output now that all that all
  // uses are gone
  if (i != node->outputs().size()) {
    node->eraseOutput(i);
  }

  // Traverse sub-blocks if necessary
  for (auto block : node->blocks()) {
    eraseTokens(block);
  }
}
} // namespace

void AnnotateEffects(std::shared_ptr<Graph>& graph) {
  AnnotateEffectsImpl impl;
  impl.annotateEffects(graph.get());
}

void UnAnnotateEffects(std::shared_ptr<Graph>& graph) {
  eraseTokens(graph->block());

  // Special handling to delete the constant "token" generator node
  for (auto node : graph->block()->nodes())
    if (node->kind() == prim::Constant && node->outputs().size() == 0) {
      node->destroy();
      break;
    }
}
} // namespace jit
} // namespace torch
