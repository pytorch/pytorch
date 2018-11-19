#include "torch/csrc/jit/passes/batch_mm.h"

#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/custom_operator.h"
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
static constexpr size_t min_fusion_size = 4;

bool have_same_shape(at::TensorList inputs) {
  auto expected_sizes = inputs[0].sizes();
  return std::all_of(inputs.begin(), inputs.end(),
                     [expected_sizes](const at::Tensor& t) {
                       return t.sizes() == expected_sizes;
                     });
}

bool shape_is_fast(const at::Tensor& lhs, const at::Tensor& rhs) {
  size_t l = lhs.size(0);
  size_t m = lhs.size(1);
  size_t r = rhs.size(1);
  // Numbers obtained by some simple benchmarks of fp32 gemms on a TITAN V
  return m < 512 || ((l < 256 && r < 256) || (l > 256 && r > 256));
}

RegisterOperators mm_tree_reduction_reg({
  Operator(
    prim::MMTreeReduce,
    [](const Node* node) {
      size_t num_inputs = node->inputs().size();
      return [num_inputs](Stack& stack) {
        std::vector<at::Tensor> inputs;
        inputs.reserve(num_inputs);
        for (auto it = stack.end() - num_inputs; it != stack.end(); ++it) {
          inputs.push_back(std::move(*it).toTensor());
        }
        drop(stack, num_inputs);

        JIT_ASSERT(inputs.size() > 0);
        JIT_ASSERT(inputs.size() % 2 == 0);
        size_t side_num_elems = inputs.size() / 2;
        auto lhs_inputs = at::TensorList(inputs).slice(0, side_num_elems);
        auto rhs_inputs = at::TensorList(inputs).slice(side_num_elems);
        // TODO: checking this is not free, so we should stop if this keeps failing
        // TODO: benchmark to find when is this really a win, and add size constraints
        if (have_same_shape(lhs_inputs) && have_same_shape(rhs_inputs) && shape_is_fast(lhs_inputs[0], rhs_inputs[0])) {
          auto lhs = at::cat(lhs_inputs, /*dim=*/1);
          auto rhs = at::cat(rhs_inputs, /*dim=*/0);
          push(stack, at::mm(lhs, rhs));
        } else {
          auto acc = at::mm(inputs[0], inputs[side_num_elems]);
          for (size_t i = 1; i < side_num_elems; ++i) {
            acc.add_(at::mm(inputs[i], inputs[side_num_elems + i]));
          }
          push(stack, std::move(acc));
        }
        return 0;
      };
    })
});

// TreeTokens will be used to label nodes of the graph, if the nodes will fit
// our mm/add tree pattern. Basically we do dynamic programming on DAGs, where
// when we reach node N with inputs A and B, then A and B have already been
// procesed, and we can try to unify their TreeTokens (if they have them)
// and build a larger tree.
struct TreeToken {
  uint64_t tree_size = 0; // NOTE: measured in number of leaves i.e. mm ops
  Node *node = nullptr;
  bool is_root = false;

  static TreeToken mm(Node *mm) {
    TreeToken token;
    token.tree_size = 1;
    token.node = mm;
    token.is_root = true;
    return token;
  }

  // NB: the returned token might be invalid, so make sure to check its boolean value!
  static TreeToken transpose(Node *t, TreeToken& inp_token) {
    TreeToken token;
    if (!inp_token.node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      return token;
    }
    token.tree_size = 1;
    token.node = t;
    token.is_root = true;
    inp_token.is_root = false;
    return token;
  }

  // NB: the returned token might be invalid, so make sure to check its boolean value!
  static TreeToken add(Node *add, TreeToken& l, TreeToken& r) {
    TreeToken token;
    // See Note [Overlapping trees]
    if (&l == &r || !l.is_root || !r.is_root)
      return token;
    token.tree_size = l.tree_size + r.tree_size;
    token.node = add;
    token.is_root = true;
    l.is_root = r.is_root = false; // Reserve the subtrees, so they can't be used again.
    return token;
  }

  explicit operator bool() {
    return is_root;
  }

  std::vector<Node*> removeTransposesAndGatherMatmuls() {
    std::vector<Node*> matmuls;
    std::vector<Node*> queue {node};
    Graph* graph = node->owningGraph();
    while (!queue.empty()) {
      auto n = queue.back(); queue.pop_back();
      if (n->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
        matmuls.push_back(n);
      } else if (n->matches("aten::t(Tensor self) -> Tensor")) {
        Node * input_node = n->input()->node();
        JIT_ASSERT(input_node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor"));
        // (AB)^T == B^TA^T
        WithInsertPoint insert_guard { input_node };
        Value * A = input_node->inputs()[0];
        Value * B = input_node->inputs()[1];
        Value * AT = graph->insert(aten::t, {A});
        Value * BT = graph->insert(aten::t, {B});
        Value * BTAT = graph->insert(aten::mm, {BT, AT});
        n->output()->replaceAllUsesWith(BTAT);
        matmuls.push_back(BTAT->node());
      } else if (n->matches("aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor")) {
        queue.push_back(n->inputs()[0]->node());
        queue.push_back(n->inputs()[1]->node());
      } else {
        AT_ASSERTM(false, "Unsupported node found in a BatchMM tree!");
      }
    }
    return matmuls;
  }
};

void BatchMMBlock(Block* block) {
  enum class Side { LHS, RHS };
  auto graph = block->owningGraph();

  // Look for trees in the block
  std::unordered_map<Node*, TreeToken> tokens;
  for (auto node : block->nodes()) {
    if (node->matches("aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
      tokens[node] = TreeToken::mm(node);
    } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
      auto input_it = tokens.find(node->input()->node());
      if (input_it != tokens.end()) {
        tokens[node] = TreeToken::transpose(node, input_it->second);
      }
    } else if (node->matches("aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor")) {
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
        if (auto token = TreeToken::add(node, lhs_it->second, rhs_it->second)) {
          tokens[node] = token;
        }
      }
    } else {
      for (auto block : node->blocks()) {
        BatchMMBlock(block);
      }
    }
  }

  // Merge trees we've found
  for (auto & item : tokens) {
    auto & root = item.second;
    if (!root || root.tree_size < min_fusion_size)
      continue;
    auto matmuls = root.removeTransposesAndGatherMatmuls();
    WithInsertPoint insert_guard {root.node};
    Node * tree_reduce = graph->insertNode(graph->create(Symbol::prim("MMTreeReduce")));
    for (Node * matmul : matmuls) {
      tree_reduce->addInput(matmul->inputs().at(0));
    }
    for (Node * matmul : matmuls) {
      tree_reduce->addInput(matmul->inputs().at(1));
    }
    root.node->output()->replaceAllUsesWith(tree_reduce->output());
    // NB: don't bother with cleaning up after yourself. We'll use DCE for that.
  }
}

void BatchMM(std::shared_ptr<Graph>& graph) {
  BatchMMBlock(graph->block());
  EliminateDeadCode(graph);
  // It's possible that transpose rearrangements have created sequences of consecutive
  // transposes that didn't exist before.
  PeepholeOptimize(graph);
}

}}
