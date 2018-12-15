#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <torch/csrc/jit/passes/alias_analysis.h>

#include <unordered_map>

namespace torch {
namespace jit {

class DeadCodeEliminator {
 public:
  explicit DeadCodeEliminator(std::shared_ptr<Graph> graph)
      : aliasDb_(AliasAnalysis(std::move(graph))) {}
  DeadCodeEliminator(bool collect_only = false)
      : collect_only_(collect_only) {}

  // The algorithm is an inverse mark-and-sweep. Starting from the return node,
  // we mark "live" nodes that are necessary for the output. Nodes that have
  // side effects are also marked.
  void run(Block* block, bool recurse) {
    // Find the last wildcard in the block. We cannot eliminate any mutable ops
    // that precede the last wildcard.
    setLastWildcard();

    // Initialize by adding the return node to work list
    markAndEnqueue(block->return_node());

    mark(block);
    sweep(block, recurse);
  }

  // *_once, because it should be called only once per run() call.
  std::unordered_set<Node*> get_dead_once() {
    JIT_ASSERT(collect_only_);
    return std::move(dead_nodes_);
  }

 private:
  void setLastWildcard() {
    if (!aliasDb_) {
      return;
    }

    const auto& wildcards = aliasDb_->getWildcardNodes();
    if (wildcards.empty()) {
      return;
    }

    lastWildcard_ = *wildcards.begin();
    for (const auto wildcard : wildcards) {
      if (wildcard->isAfter(*lastWildcard_)) {
        lastWildcard_ = wildcard;
      }
    }
  }

  void mark(Block* block) {
    // Mark all nodes with side effects.
    for (auto node : block->nodes()) {
      if (hasSideEffects(node)) {
        markAndEnqueue(node);
      }
    }

    while (!workQueue_.empty()) {
      auto node = workQueue_.front();
      workQueue_.pop_front();

      for (auto subBlock : node->blocks()) {
        mark(subBlock);
      }

      // Mark all nodes in this node's blockchain (since owning nodes are
      // considered live if they contain a live node)
      if (node->owningBlock() != block) {
        auto curNode = node;
        while (curNode) {
          if (!curNode->owningBlock()) {
            break;
          }

          markAndEnqueue(curNode);
          curNode = curNode->owningBlock()->owningNode();
        }
      }

      // Find preceding writers for node, add to work list
      if (aliasDb_) {
        for (auto writer : aliasDb_->getWritersForNode(node)) {
          if (writer->isBefore(node)) {
            markAndEnqueue(writer);
          }
        }
      }

      // Find producers for all inputs, add to work list
      for (auto input : node->inputs()) {
        markAndEnqueue(input->node());
      }
    }
  }

  // Delete all unmarked nodes.
  void sweep(Block* block, bool recurse) {
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      // note these occur before the recursion because we want to uncover
      // dead code in the blocks used to calculate the output
      if (!collect_only_) {
        removeDeadIfOutputs(node);
        removeDeadLoopOutputs(node);
      }
      if (recurse) {
        for (Block* block : node->blocks()) {
          sweep(block, true);
        }
      }
      // TODO(suo): We shouldn't really have to check whether a node has uses,
      // since the mark algorithm should do that. But currently, the marking
      // doesn't reach loop counters in certain cases (see TestScript.test_pass)
      if (!marked_.count(node) && !hasUsesOutsideDeadNodes(node)) {
        if (collect_only_) {
          dead_nodes_.insert(node);
        } else {
          it.destroyCurrent();
        }
      }
    }
  }

  bool hasUsesOutsideDeadNodes(Node * n) {
    if (!collect_only_) {
      return n->hasUses();
    }
    for (Value * output : n->outputs()) {
      for (const Use & u : output->uses()) {
        if (dead_nodes_.count(u.user) == 0) {
          return true;
        }
      }
    }
    return false;
  }

  void markAndEnqueue(Node* n) {
    if (!marked_.count(n)) {
      marked_.insert(n);
      workQueue_.push_back(n);
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
      // Otherwise, there are two kinds of nodes with untracked effects:
      // 1. Nodes that write to a value that may alias the graph inputs (since
      //    the inputs can be used outside the graph).
      // 2. Anything that could clobber a wildcard value.
      bool touchesWildcard = false;
      if (lastWildcard_) {
        touchesWildcard = aliasDb_->hasWrites(node) &&
            (node->isBefore(*lastWildcard_) || node == *lastWildcard_);
      }
      return aliasDb_->writesToInputAlias(node) || touchesWildcard;
    }
  }

  bool hasSideEffects(Node* node) {
    auto it = memo_.find(node);
    if (it != memo_.end())
      return it->second;
    bool has_side_effects = node->kind() == prim::Print ||
        node->kind() == aten::warn ||
        node->kind() == prim::RaiseException ||
        node->kind() == prim::PythonOp ||
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

  void removeDeadIfOutputs(Node* node) {
    if (node->kind() != prim::If)
      return;

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

  c10::optional<AliasDb> aliasDb_;
  std::unordered_map<Node*, bool> memo_;
  std::unordered_set<Node*> marked_;
  std::list<Node*> workQueue_;
  c10::optional<const Node*> lastWildcard_;

  bool collect_only_ = false;
  std::unordered_set<Node*> dead_nodes_; // Will be filled iff collect_only_ is true
};

void EliminateDeadCode(const std::shared_ptr<Graph>& graph) {
  DeadCodeEliminator(graph).run(graph->block(), /*recurse=*/true);
}

void EliminateDeadCode(Block* block, bool recurse) {
  DeadCodeEliminator().run(block, recurse);
}

std::unordered_set<Node*> FindDeadNodes(Block* block, bool recurse) {
  DeadCodeEliminator eliminator(/*collect_only=*/true);
  eliminator.run(block, recurse);
  return eliminator.get_dead_once();
}

} // namespace jit
} // namespace torch
