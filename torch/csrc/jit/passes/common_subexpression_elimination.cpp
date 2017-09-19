#include "torch/csrc/jit/ir.h"

#include <algorithm>
#include <unordered_map>

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

namespace torch { namespace jit {

struct HashNodeCSE {
  std::size_t operator()(const Node* k) const {
    JIT_ASSERT(k != nullptr);
    std::size_t p = 31; // A prime.
    std::size_t h = k->kind() * p + k->stage();
    for (auto i : k->inputs()) {
      h = h * p + i->unique();
    }
    return h;
  }
};

struct EqualNodeCSE {
  bool operator()(const Node* lhs, const Node* rhs) const {
    if (lhs == nullptr && rhs == nullptr) return true;
    if (lhs == nullptr || rhs == nullptr) return false;

    // Check whether two nodes are the same kind.
    if (lhs->kind() != rhs->kind()) return false;

    // Check the stage.
    if (lhs->stage() != rhs->stage()) return false;

    // TODO check the device.

    // Check whether the inputs are the same.
    if (lhs->inputs().size() != rhs->inputs().size()) return false;

    if (!std::equal(lhs->inputs().begin(), lhs->inputs().end(), rhs->inputs().begin())) return false;

    // Check the attributes.
    // TODO support attributes comparison.
    if (lhs->hasAttributes() || rhs->hasAttributes()) return false;

    return true;
  }
};

void EliminateCommonSubexpression(std::shared_ptr<Graph>& graph) {
  // Keep iterating until reach the fixed point.
  bool reach_fixed = false;
  while (!reach_fixed) {
    reach_fixed = true;
    auto nodes = graph->nodes();
    std::unordered_set<Node*, HashNodeCSE, EqualNodeCSE> subexprs;
    for (auto it = nodes.begin(); it != nodes.end(); ++ it) {
      auto node = *it;
      if (node->kind() != kAdd
          && node->kind() != kMul
          && node->kind() != kNeg
          && node->kind() != kSigmoid
          && node->kind() != kTanh
          && node->kind() != kSplit
          && node->kind() != kAddConstant
         ) {
        // TODO support more kinds of nodes.
        // Only support CSE on these nodes.
        continue;
      }

      // Check whether the same subexpression already exists.
      if (subexprs.find(node) == subexprs.end()) {
        // If not put it into the map
        subexprs.insert(node);
      } else {
        // Subexpression exists, replace the uses of node, and destroy it.
        auto existing = *subexprs.find(node);
        const use_list & uses = node->uses();
        const use_list & reuses= existing->uses();
        if (node->hasMultipleOutputs()) {
          // For Multi-Output nodes, all its uses should be Select nodes.
          JIT_ASSERT(uses.size() == reuses.size());
          // Replace the uses of Select nodes.
          for (size_t i = 0; i < uses.size(); ++ i) {
            JIT_ASSERT(uses[i].user->kind() == kSelect);
            JIT_ASSERT(reuses[i].user->kind() == kSelect);
            uses[i].user->replaceAllUsesWith(reuses[i].user);
          }
          // Destroy Select nodes.
          while (uses.size() > 0) {
            uses[0].user->destroy();
          }
        } else {
          node->replaceAllUsesWith(existing);
        }
        // Destroy the node.
        node->destroy();
        reach_fixed = false;
        break;
      }
    }
  }
}

}}
