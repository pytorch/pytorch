#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>

#include <c10/util/hash.h>
#include <unordered_map>

namespace torch {
namespace jit {
namespace {

// There are a context managers which change global state -
// with torch.no_grad(), with torch.cpu.amp.autocast
// These are represented in JIT as prim::Enter and prim::Exit nodes
// Avoid CSE across two separate with statements

struct NodeAndContextNode : public std::pair<Node*, Node*> {
  using pair::pair;

  Node* node() const {
    return this->first;
  }
  Node* contextNode() const {
    return this->second;
  }
};

struct TORCH_API HashNodeAndContext {
  size_t operator()(const NodeAndContextNode pair) const {
    HashNode hash;
    // we hash on the properties of the Node to be CSE'd, and the exact node
    // of the context its in (which may be a nullptr)
    return c10::hash_combine(
        hash(pair.node()), reinterpret_cast<size_t>(pair.contextNode()));
  }
};

struct TORCH_API EqualNodeAndContext {
  bool operator()(const NodeAndContextNode lhs, const NodeAndContextNode rhs)
      const {
    EqualNode eq;
    bool nodes_equal = eq(lhs.node(), rhs.node());
    //  similarly to equality, check equality of properties of the nodes to be
    //  CSE'd
    // and the exact node of its context (which may be nullptr)
    return nodes_equal && (lhs.contextNode() == rhs.contextNode());
  }
};

struct CommonSubexpressionEliminator {
  CommonSubexpressionEliminator(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run(std::function<Node*(Node*)> parent_lookup_fn) {
    return run(graph_->block(), std::move(parent_lookup_fn));
  }

  // The function implements common subexpression elimination.
  // Since the nodes are visited in topological order, one pass is enough.
  // returns true if CSE made changes to a graph
  bool run(Block* block, std::function<Node*(Node*)> parent_lookup_fn) {
    std::unordered_set<
        NodeAndContextNode,
        HashNodeAndContext,
        EqualNodeAndContext>
        subexprs;
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto node = *it;

      if (node->kind() == prim::Enter) {
        prim_enter_stack_.push_back(node);
      }
      if (node->kind() == prim::Exit) {
        prim_enter_stack_.pop_back();
      }

      if (node->kind() == prim::profile) {
        GRAPH_DEBUG(
            "Profiled nodes shouldn't be CSE'ed there's a separate pass that does dedup and merging:\n",
            *node);
        continue;
      }

      if (node->hasSideEffects()) {
        GRAPH_DEBUG("Node was skipped due to side effects:\n", *node);
        continue;
      }
      if (node->isNondeterministic()) {
        GRAPH_DEBUG("Node was skipped due to its non determinism:\n", *node);
        continue;
      }

      if (!node->blocks().empty()) {
        // Traverse sub-blocks.
        for (auto block : node->blocks()) {
          changed |= run(block, [&](Node* n) {
            NodeAndContextNode nacn(
                n,
                prim_enter_stack_.size() == 0 ? nullptr
                                             : prim_enter_stack_.back());
            auto existing = subexprs.find(nacn);
            if (existing != subexprs.end()) {
              return (*existing).node();
            }

            return parent_lookup_fn(n);
          });
        }

        continue;
      }

      if (getOrCreateAliasDb().hasWriters(node)) {
        GRAPH_DEBUG("Node was skipped due to alias analysis result:\n", *node);
        // Do NOT have enough information to do CSE on these nodes.
        continue;
      }

      // Check for CSE opportunities in the parent block.
      auto parent_lookup = parent_lookup_fn(node);
      auto g_out = node->owningGraph()->outputs();
      if (parent_lookup != nullptr) {
        if (!getOrCreateAliasDb().safeToChangeAliasingRelationship(
                node->outputs(), parent_lookup->outputs())) {
          continue;
        }

        GRAPH_UPDATE("Replacing\n", *node, "with\n", *parent_lookup);
        changed = true;
        node->replaceAllUsesWith(parent_lookup);
        it.destroyCurrent();
        continue;
      }

      // Check whether the same subexpression already exists.
      Node* enter_node =
          prim_enter_stack_.size() == 0 ? nullptr : prim_enter_stack_.back();
      auto subit = subexprs.insert(NodeAndContextNode(node, enter_node));
      if (!subit.second) {
        // Subexpression exists, replace the uses of node, and destroy it.
        auto existing = (*subit.first).node();

        // don't introduce new aliasing among graph outputs
        if (getOrCreateAliasDb().mayContainAlias(
                node->outputs(), node->owningGraph()->outputs()) &&
            getOrCreateAliasDb().mayContainAlias(existing->outputs(), g_out)) {
          continue;
        }

        GRAPH_UPDATE("Replacing\n", *node, "with\n", *existing);
        changed = true;
        node->replaceAllUsesWith(existing);
        // Destroy the node.
        it.destroyCurrent();
      }
    }

    return changed;
  }

  AliasDb& getOrCreateAliasDb() {
    if (!alias_db_) {
      alias_db_ = std::make_unique<AliasDb>(graph_);
    }

    return *alias_db_;
  }

 private:
  std::vector<Node*> prim_enter_stack_;
  std::unique_ptr<AliasDb> alias_db_;
  std::shared_ptr<Graph> graph_;
};

} // namespace

bool EliminateCommonSubexpression(const std::shared_ptr<Graph>& graph) {
  GRAPH_DUMP("Before CSE", graph);
  CommonSubexpressionEliminator cse(graph);
  return cse.run([](Node*) { return nullptr; });
}
} // namespace jit
} // namespace torch
