#include <torch/csrc/jit/passes/canonicalize.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir_views.h>

namespace torch {
namespace jit {

// Canonicalize a graph, renumbering it so that all structurally equivalent
// graphs have same numbers.
// keep_unique_names: If false, canonicalizes unique names by removing them
//   and replacing them with normal value names.
//   Otherwise, ignores values with unique names.
std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names) {
  auto r = std::make_shared<Graph>(graph->current_scope());
  std::unordered_map<Value*, Value*> rn_env;
  auto rn_fn = [&](Value* v) { return rn_env.at(v); };
  for (auto* input : graph->inputs()) {
    auto* r_input = r->addInput();
    r_input->copyMetadata(input);
    if (!keep_unique_names)
      r_input->setDebugName("");
    rn_env[input] = r_input;
  }
  for (auto* node : graph->nodes()) {
    auto* r_node = r->createClone(node, rn_fn);
    if (!keep_unique_names) {
      for (auto* output : r_node->outputs()) {
        output->setDebugName("");
      }
    }
    r->appendNode(r_node);
    auto outputs = node->outputs();
    auto r_outputs = r_node->outputs();
    for (const auto i : c10::irange(outputs.size())) {
      rn_env[outputs.at(i)] = r_outputs.at(i);
    }
    if (node->hasAttribute(attr::Subgraph)) {
      r_node->g_(
          attr::Subgraph,
          Canonicalize(node->g(attr::Subgraph), keep_unique_names));
    }
  }
  for (auto* output : graph->outputs()) {
    r->registerOutput(rn_fn(output));
  }

  return r;
}

// Which index in b's owning Node is b
static size_t blockIndex(const Block* b) {
  auto n = b->owningNode();
  AT_ASSERT(n);
  for (size_t i = 0; i < n->blocks().size(); ++i) {
    if (n->blocks()[i] == b) {
      return i;
    }
  }
  AT_ASSERT(false);
}

/*
 * This establishes a canonical ordering of nodes.
 * If n1 and n2 are in the same block, whichever node appears first
 * is before the other.
 * If n1 and n2 are contained in different blocks of an if node,
 * then whichever block is in the true block is ordered before the other.
 * If n1 contains n2, then n1 is before n2. This has the nice property that
 * whichever node appears first in a dump of the graph is before the other.
 * NB: this is not a topological index. Topologically, two nodes in
 * different blocks of an if node are not topologically < or > each other.
 */
static bool isBefore(Node* n1, Node* n2) {
  // Invalid to call with the same node as both args
  AT_ASSERT(n1 != n2);

  // Set n1 and n2 to be the number of blocks from the Graph block
  size_t d_1 = n1->blocksFromGraphBlock();
  size_t d_2 = n2->blocksFromGraphBlock();

  for (; d_1 > d_2; --d_1) {
    n1 = n1->owningBlock()->owningNode();
    // n2 contains n1
    if (n1 == n2) {
      return false;
    }
  }

  for (; d_2 > d_1; --d_2) {
    n2 = n2->owningBlock()->owningNode();
    // n1 contains n2
    if (n2 == n1) {
      return true;
    }
  }

  // Now they are the same numer of blocks from the graph block,
  // recurse upwards, checking if they are on the same block
  while (true) {
    if (n1->owningBlock() == n2->owningBlock()) {
      return n1->isBefore(n2);
    }

    auto new_n1 = n1->owningBlock()->owningNode();
    auto new_n2 = n2->owningBlock()->owningNode();

    AT_ASSERT(new_n1 != nullptr);
    AT_ASSERT(new_n2 != nullptr);

    if (new_n1 == new_n2) {
      // take whichever node is in the earlier block
      auto index_1 = blockIndex(n1->owningBlock());
      auto index_2 = blockIndex(n2->owningBlock());
      return index_1 < index_2;
    }

    n1 = new_n1;
    n2 = new_n2;
  }
}

static bool isBefore(const Use& a, const Use& b) {
  // If two uses are the same node, we order on offset
  if (a.user == b.user) {
    return a.offset < b.offset;
  }

  return isBefore(a.user, b.user);
}

static bool isAfter(const Use& a, const Use& b) {
  if (a.user == b.user && a.offset == b.offset) {
    return false;
  }
  return !isBefore(a, b);
}

bool isBeforeOrAfter(const Use& a, const Use& b, bool checking_before) {
  return checking_before ? isBefore(a, b) : isAfter(a, b);
}

std::optional<const Use> firstOrLastUse(Value* v, bool find_first) {
  if (v->uses().empty()) {
    return c10::nullopt;
  }
  Use extreme_use = v->uses()[0];
  for (size_t i = 1; i < v->uses().size(); ++i) {
    auto n_use = v->uses()[i];
    if (!isBeforeOrAfter(extreme_use, n_use, find_first)) {
      extreme_use = n_use;
    }
  }

  return extreme_use;
}

static std::vector<std::optional<const Use>> gatherFirstUses(
    at::ArrayRef<Value*> values) {
  return fmap(values, [&](Value* v) -> std::optional<const Use> {
    return firstOrLastUse(v, true);
  });
}

static std::vector<size_t> sort_indexes(at::ArrayRef<Value*> values) {
  // initialize original index locations
  std::vector<size_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::vector<std::optional<const Use>> first_uses = gatherFirstUses(values);

  // Sort values based on canonical ordering of their first usage
  std::sort(idx.begin(), idx.end(), [&first_uses](size_t i1, size_t i2) {
    // if neither has any uses, use original ordering. Since the
    // only values that jitter are ones added by the compiler and are guaranteed
    // to have uses, original ordering is fine.
    if (first_uses[i1] == c10::nullopt && first_uses[i2] == c10::nullopt) {
      return i1 < i2;
    }
    if (first_uses[i1] == c10::nullopt) {
      return false;
    } else if (first_uses[i2] == c10::nullopt) {
      return true;
    }

    auto fst_v1 = *first_uses[i1];
    auto fst_v2 = *first_uses[i2];

    return isBefore(fst_v1, fst_v2);
  });

  return idx;
}

static void CanonicalizeLoopOutputs(Node* n) {
  auto new_indices = sort_indexes(n->outputs());
  LoopView(n).permuteLoopCarried(new_indices);
}

static void CanonicalizeIfOutputs(Node* n) {
  auto new_indices = sort_indexes(n->outputs());
  IfView(n).permuteOutputs(new_indices);
}

static void CanonicalizeOutputs(Block* block) {
  // We iterate in reverse since ordering of a node's outputs is dependent on
  // the value use following it in the graph
  for (Node* n : block->nodes().reverse()) {
    switch (n->kind()) {
      case prim::Loop: {
        CanonicalizeLoopOutputs(n);
      } break;
      case prim::If: {
        CanonicalizeIfOutputs(n);
      } break;
    }
    // Since an a control flow node's outputs are after
    // the values outputted within its blocks, first canonicalize
    // the nodes outputs and then recurse on its blocks
    for (Block* b : n->blocks()) {
      CanonicalizeOutputs(b);
    }
  }
}

// Canonicalize a graph's control flow node outputs. We do this to solve jitter
// issues with outputs added to control flow nodes after the first pass of
// compilation in ir_emitter.cpp
void CanonicalizeOutputs(std::shared_ptr<Graph>& graph) {
  CanonicalizeOutputs(graph->block());
}
} // namespace jit
} // namespace torch
