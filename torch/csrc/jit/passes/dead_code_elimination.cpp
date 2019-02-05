#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/utils/memory.h>

#include <unordered_map>

namespace torch {
namespace jit {

namespace prim {
using namespace ::c10::prim;
}

class DeadCodeEliminator {
 public:
  explicit DeadCodeEliminator(std::shared_ptr<Graph> graph)
      : aliasDb_(torch::make_unique<AliasDb>(std::move(graph))) {}
  DeadCodeEliminator() = default;

  // The algorithm is an inverse mark-and-sweep. Starting from the return node,
  // we mark "live" nodes that are necessary for the output. Nodes that have
  // side effects are also marked.
  void run(Block* block, bool recurse) {
    // Initialize by marking the return node and all its consumed values as live
    mark(block->return_node());

    mark(block);

    deleteCallback_(liveValues_);

    sweep(block, recurse);
  }

  void setDeleteCallback(
      std::function<void(const std::unordered_set<const Value*>&)>
          deleteCallback) {
    deleteCallback_ = std::move(deleteCallback);
  }

 private:
  // Special handling for block return nodes. Unlike other nodes, the block
  // return node doesn't really "use" its inputs. Consider:
  //
  // %a0 = aten::foo()
  // %b = aten::foo()
  // %a2, %b2 = prim::If(%cond) {
  //   block0() {
  //     %a1 = aten::foo(%.0)
  //     %b1 = aten::foo(%b)
  //   } -> (%a1, %b1)
  // }
  // return (%a2)
  //
  // We want to be able to DCE all the %b stuff. So when processing block
  // returns, we only mark producers for values that "live" (i.e. used outside
  // the block).
  void markReturnNode(Node* node) {
    if (marked_.count(node)) {
      return;
    }

    AT_ASSERT(node->owningBlock()->return_node() == node);
    auto outerNode = node->owningBlock()->owningNode();
    if (outerNode == nullptr || outerNode->kind() == prim::Reverse) {
      // If there's no outer node, we're looking at the graph's top-level
      // return block. We consider all graph outputs to be "used", so just mark
      // this node normally.
      return mark(node);
    }

    // Collect all inputs that are actually live
    if (outerNode->kind() == prim::Loop ||
        outerNode->kind() == c10::onnx::Loop) {
      // Special handling to deal with loop carried dependencies.
      auto loop = LoopView(outerNode);
      for (size_t i = 0; i < loop.carriedOutputs().size(); i++) {
        auto innerInput = loop.bodyCarriedInputs().at(i);
        auto innerOutput = loop.bodyCarriedOutputs().at(i);
        auto outerOutput = loop.carriedOutputs().at(i);
        if (liveValues_.count(outerOutput) || innerInput->hasUses()) {
          liveValues_.insert(innerOutput);
        }
      }

      // Also mark the loop next condition as live, since it will be used inside
      // the loop body.
      liveValues_.insert(loop.nextCond());
    } else {
      AT_ASSERT(outerNode->outputs().size() == node->inputs().size());
      for (size_t i = 0; i < outerNode->outputs().size(); i++) {
        auto innerOutput = node->inputs()[i];
        auto outerOutput = outerNode->outputs()[i];
        if (liveValues_.count(outerOutput)) {
          liveValues_.insert(innerOutput);
        }
      }
    }

    marked_.insert(node);
  }

  void mark(Block* block) {
    // Mark all nodes with side effects.
    for (auto node : block->nodes()) {
      if (hasSideEffects(node)) {
        mark(node);
      }
    }

    // Initialize by marking the return node
    markReturnNode(block->return_node());

    for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); ++it) {
      auto node = *it;
      for (auto subBlock : node->blocks()) {
        mark(subBlock);
      }
      markIfLive(node);
    }
  }

  // If we output or write to a live memory location, mark this node
  void markIfLive(Node* node) {
    for (const auto output : node->outputs()) {
      if (liveValues_.count(output)) {
        return mark(node);
      }
    }

    if (aliasDb_) {
      const auto writes = aliasDb_->getWrites(node);
      if (aliasDb_->mayAlias(writes, liveValues_)) {
        return mark(node);
      }
    }
  }

  // Mark this node as live and add this node's inputs and aliases to the live
  // value sets.
  void mark(Node* node) {
    if (marked_.count(node)) {
      return;
    }

    marked_.insert(node);

    // Mark all nodes in this node's blockchain (since owning nodes are
    // considered live if they contain a live node)
    auto curNode = node;
    while (curNode) {
      if (!curNode->owningBlock()) {
        break;
      }

      mark(curNode);
      curNode = curNode->owningBlock()->owningNode();
    }

    for (const auto input : node->inputs()) {
      if (liveValues_.count(input)) {
        continue;
      }
      liveValues_.insert(input);
    }
  }

  // Delete all unmarked nodes.
  void sweep(Block* block, bool recurse) {
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      // note these occur before the recursion because we want to uncover
      // dead code in the blocks used to calculate the output
      removeDeadBlockOutputs(node);
      removeDeadLoopOutputs(node);
      if (recurse) {
        for (Block* block : node->blocks()) {
          sweep(block, true);
        }
      }
      // NB: Checking hasUses() is required. AD graphs are not perfectly
      // valid, as a node in grad_desc.f might be used in reverse_block.
      // Reverse_block is inlined in grad_desc.f before it's separated
      // to grad_desc.df.
      if (!(marked_.count(node) || node->hasUses())) {
        it.destroyCurrent();
      }
    }
  }

  bool hasUntrackedMutation(Node* node) {
    if (!aliasDb_) {
      // If we don't have alias information, all mutable ops have unknown
      // effects and can't be considered for elimination.
      if (!node->kind().is_aten()) {
        return false;
      }
      // onnx export calls EliminateDeadCode but sometimes passes invalid
      // aten operators. So we call maybeSchema so we handle the cases when
      // there is no valid schema for a node
      auto schema = node->maybeSchema();
      return schema && schema->is_mutable();
    } else {
      return aliasDb_->hasUntrackedEffects(node);
    }
  }

  bool hasSideEffects(Node* node) {
    auto it = memo_.find(node);
    if (it != memo_.end())
      return it->second;
    bool has_side_effects = node->hasSideEffects() ||
        std::any_of(node->blocks().begin(),
                    node->blocks().end(),
                    [&](Block* b) {
                      return std::any_of(
                          b->nodes().begin(), b->nodes().end(), [&](Node* n) {
                            return hasSideEffects(n);
                          });
                    }) ||
        hasUntrackedMutation(node);

    memo_.emplace(node, has_side_effects);
    return has_side_effects;
  }

  void removeDeadBlockOutputs(Node* node) {
    if (node->kind() != prim::If && node->kind() != prim::GradOf) {
      return;
    }

    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses()) {
        node->eraseOutput(i);
        for (Block* b : node->blocks()) {
          b->eraseOutput(i);
        }
      }
    }
  }

  void removeDeadLoopOutputs(Node* node) {
    if (node->kind() != prim::Loop)
      return;
    auto loop_body = node->blocks().at(0);
    auto loop_input_offset = 2; // offset of loop carried deps in input list
    auto loop_body_offset =
        1; // offset to the loop carried dependencies in block inputs/outputs

    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses() &&
          !loop_body->inputs().at(loop_body_offset + i)->hasUses()) {
        node->eraseOutput(i);
        node->removeInput(loop_input_offset + i);
        loop_body->eraseInput(loop_body_offset + i);
        loop_body->eraseOutput(loop_body_offset + i);
      }
    }
  }

  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::unordered_map<Node*, bool> memo_;
  std::unordered_set<Node*> marked_;
  std::unordered_set<const Value*> liveValues_;
  std::function<void(const std::unordered_set<const Value*>&)> deleteCallback_ =
      [](const std::unordered_set<const Value*>&) {};
};

void EliminateDeadCode(const std::shared_ptr<Graph>& graph) {
  DeadCodeEliminator(graph).run(graph->block(), /*recurse=*/true);
}

void EliminateDeadCode(Block* block, bool recurse) {
  DeadCodeEliminator().run(block, recurse);
}

void EliminateDeadCode(
    Block* block,
    std::function<void(const std::unordered_set<const Value*>&)> cb) {
  DeadCodeEliminator eliminator;
  eliminator.setDeleteCallback(std::move(cb));
  eliminator.run(block, /*recurse=*/true);
}

} // namespace jit
} // namespace torch
