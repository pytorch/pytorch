#include "torch/csrc/jit/passes/batch_mm.h"

#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/utils/functional.h"

#include <ATen/ATen.h>
#include <algorithm>
#include <unordered_map>

namespace torch { namespace jit {

// This pass looks for trees in the graph, where leaves are mm ops, and the inner
// vertices are add nodes. Once we have such a tree they can be reduced to two
// concats and a single mm (basically into a single multiply of a wide matrix, with
// a tall matrix).
// Such patterns show up mostly in backward of RNNs, since the derivative of many
// uses of matrix multiplies with same weights forms exactly such a tree
// (note that it's usually also highly imbalanced i.e. has O(n) depth).
//
// This (or any tree of adds of MMs):
//
// +------+ +------+   +------+ +------+   +------+
// |      | |      |   |      | |      |   |      |
// |  L1  | |  R1  | + |  L2  | |  R2  | = |  O   |
// |      | |      |   |      | |      |   |      |
// +------+ +------+   +------+ +------+   +------+
//
// can be basically transformed into a single MM which looks like this
// (we concat all lhs operands, concat rhs operands, do mm):
//
//                 +------+
//                 |      |
//                 |  R1  |
//                 |      |
//                 +------+
//                 |      |
//                 |  R2  |
//                 |      |
//                 +------+
// +------+------+ +------+
// |      |      | |      |
// |  L1  |  L2  | |  O   |
// |      |      | |      |
// +------+------+ +------+

// Note [Further optimizations]
// It would be straightforward to extend the TreeToken class to also detect if all
// MMs had the same lhs/rhs. In such case it's more efficient to expand the lhs
// and use bmm + sum instead of repeating it in memory via concat.

// Note [Overlapping trees]
// Additionally it wouldn't be too hard to add support for partially overlapping
// trees. Right now the it's forbidden in the algorithm (only a single tree will
// be allowed), so theoretically we might miss some optimization options, especially
// that the rejected tree could be much larger. I didn't implement that because it's
// not necessary for the simple RNN cases I saw, so I decided to keep stuff simple.
// If we ever get around implementing this, the right solution is probably to fuse
// MMs for the common part, and assume it's an input leaf for the outer two parts
// (I don't think it's beneficial to recompute, unless the subtree is super small,
// but let's not get into such details).

// The algorithm we're using is simple. We're iterating through the graph in the
// topological order and labeling nodes with TreeTokens. Then, we look for roots of
// the trees we formed and fuse them.

// Tunable parameter. Set to something larger if it turns out to be better.
static constexpr std::size_t min_fusion_size = 2;

static std::array<int64_t, 2> as_array(at::IntList sizes) {
  JIT_ASSERT(sizes.size() == 2);
  std::array<int64_t, 2> arr;
  arr[0] = sizes[0];
  arr[1] = sizes[1];
  return arr;
}

// TreeTokens will be used to label nodes of the graph, if the nodes will fit
// our mm/add tree pattern. Basically we do dynamic programming on DAGs, where
// when we reach node N with inputs A and B, then A and B have already been
// procesed, and we can try to unify their TreeTokens (if they have them)
// and build a larger tree.
struct TreeToken {
  uint64_t tree_size = 0; // NOTE: measured in number of leaves i.e. mm ops
  std::array<int64_t, 2> lhs_sizes;
  std::array<int64_t, 2> rhs_sizes;
  Node *node = nullptr;
  bool is_root = false;

  static TreeToken fromMM(Node *mm) {
    TreeToken token;
    token.tree_size = 1;
    Value *lhs = mm->inputs()[0];
    Value *rhs = mm->inputs()[1];
    token.lhs_sizes = as_array(lhs->type()->expect<TensorType>()->sizes());
    token.rhs_sizes = as_array(rhs->type()->expect<TensorType>()->sizes());
    token.node = mm;
    token.is_root = true;
    return token;
  }

  static TreeToken unify(Node *add, TreeToken& l, TreeToken& r) {
    TreeToken token;
    // See Note [Overlapping trees]
    if (&l == &r || !l.is_root || !r.is_root)
      return token;
    // We can batch the tree only if all sizes match, because we need to
    // cat inputs for both operands
    if (l.lhs_sizes != r.lhs_sizes)
      return token;
    if (l.rhs_sizes != r.rhs_sizes)
      return token;
    token.tree_size = l.tree_size + r.tree_size;
    token.lhs_sizes = l.lhs_sizes;
    token.rhs_sizes = l.rhs_sizes;
    token.node = add;
    token.is_root = true;
    l.is_root = r.is_root = false; // Reserve the subtrees, so they can't be used again.
    return token;
  }

  operator bool() {
    return is_root;
  }

  std::vector<Node*> gatherMatMuls() {
    static const Symbol mm_kind = "mm"_sym;
    std::vector<Node*> matmuls;
    std::vector<Node*> queue {node};
    while (!queue.empty()) {
      auto n = queue.back(); queue.pop_back();
      if (n->kind() == mm_kind) {
        matmuls.push_back(n);
      } else {
        queue.push_back(n->inputs()[0]->node());
        queue.push_back(n->inputs()[1]->node());
      }
    }
    return matmuls;
  }
};

void BatchMM(std::shared_ptr<Graph>& graph) {
  enum class Side { LHS, RHS };
  static const Symbol mm_kind = "mm"_sym;
  static const Symbol add_kind = "add"_sym;
  static const Symbol cat_kind = "cat"_sym;
  static const Symbol dim_sym = "dim"_sym;

  // Look for trees in the graph
  std::unordered_map<Node*, TreeToken> tokens;
  for (auto node : graph->nodes()) {
    if (node->kind() == mm_kind) {
      tokens[node] = TreeToken::fromMM(node);
    } else if (node->kind() == add_kind) {
      // NOTE: x + 2 is add[other={2}](%x)
      if (node->inputs().size() != 2) continue;
      Node *lhs = node->inputs()[0]->node();
      Node *rhs = node->inputs()[1]->node();
      auto lhs_it = tokens.find(lhs);
      auto rhs_it = tokens.find(rhs);
      // See Note [Overlapping trees] (regarding the uses().size() == 1 check)
      // We could treat a subtree with multiple uses as if it was overlapping.
      // XXX: uses().size() == 1 is also something that guarantees that this
      // transform is valid, because we know for sure that the none of these
      // operands depend on the result of the other. If we were to remove this,
      // we need to compute a transitive closure and actually check the dependencies.
      if (lhs_it != tokens.end() && rhs_it != tokens.end() &&
          lhs->output()->uses().size() == 1 && rhs->output()->uses().size() == 1) {
        if (auto token = TreeToken::unify(node, lhs_it->second, rhs_it->second))
          tokens[node] = token;
      }
    }
  }

  // Merge trees we've found
  for (auto & item : tokens) {
    auto & root = item.second;
    if (!root || root.tree_size < min_fusion_size)
      continue;
    auto matmuls = root.gatherMatMuls();
    auto type = root.node->output()->type()->expect<TensorType>();

    auto batch_inputs = [&](Side s, std::array<int64_t, 2> cat_sizes) -> Value* {
      int inputs_off = s == Side::LHS ? 0 : 1;
      int cat_dim    = s == Side::LHS ? 1 : 0;
      cat_sizes[cat_dim] *= matmuls.size(); // make them really cat_sizes

      auto inputs = fmap(matmuls, [=](Node *mm) { return mm->inputs()[inputs_off]; });
      Node *cat = graph->create(cat_kind, inputs)
                       ->i_(dim_sym, cat_dim);
      cat->insertBefore(root.node);
      cat->output()->setType(type->withSizes(cat_sizes));
      return cat->output();
    };

    auto lhs_batch = batch_inputs(Side::LHS, root.lhs_sizes);
    auto rhs_batch = batch_inputs(Side::RHS, root.rhs_sizes);
    Node *batch_mm = graph->create(mm_kind, {lhs_batch, rhs_batch});
    batch_mm->output()->setType(type->asShared());
    batch_mm->insertBefore(root.node);
    root.node->output()->replaceAllUsesWith(batch_mm->output());
    // NB: don't bother with cleaning up after yourself. We'll use DCE for that.
  }
  EliminateDeadCode(graph);
}

}}
