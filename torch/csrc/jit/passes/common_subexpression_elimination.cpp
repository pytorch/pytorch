#include "torch/csrc/jit/ir.h"

#include <algorithm>
#include <unordered_map>

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

namespace torch { namespace jit {



// Check whether two nodes have the same attributes in CSE.
// This function may be too conservative for general use.
bool attributesEqualCSE(const Node* lhs, const Node* rhs) {
  JIT_ASSERT(lhs != nullptr);
  JIT_ASSERT(rhs != nullptr);
  if (lhs->hasAttributes() && rhs->hasAttributes()) return true;
  if (lhs->hasAttributes() || rhs->hasAttributes()) return false;

  auto lnames = lhs->attributeNames();
  auto rnames = rhs->attributeNames();
  if (lnames != rnames) return false;

  Node* l = const_cast<Node*>(lhs);
  Node* r = const_cast<Node*>(rhs);
  for (auto name : lnames) {
    switch(l->kindOf(name)) {
      case AttributeKind::f: 
        {
          auto lv = l->f(name);
          auto rv = r->f(name);
          if (lv != rv) return false;
        }
        break;
      case AttributeKind::fs:
        {
          auto lv = l->fs(name);
          auto rv = r->fs(name);
          if (lv != rv) return false;
        }
        break;
      case AttributeKind::i:
        {
          auto lv = l->i(name);
          auto rv = r->i(name);
          if (lv != rv) return false;
        }
        break;
      case AttributeKind::is:
        {
          auto lv = l->is(name);
          auto rv = r->is(name);
          if (lv != rv) return false;
        }
        break;
      case AttributeKind::s:
        {
          auto lv = l->s(name);
          auto rv = r->s(name);
          if (lv != rv) return false;
        }
        break;
      case AttributeKind::ss:
        {
          auto lv = l->ss(name);
          auto rv = r->ss(name);
          if (lv != rv) return false;
        }
        break;
      default:
        return false;
    }
  }
  return true;
}

// Later, if someone wants to reuse this, it can be moved to some header files.
inline void hash_combine(size_t& seed, size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct HashNodeCSE {
  size_t operator()(const Node* k) const {
    JIT_ASSERT(k != nullptr);
    size_t seed = 0;
    hash_combine(seed, k->kind());
    hash_combine(seed, k->stage());
    for (auto i : k->inputs()) {
      hash_combine(seed, i->unique());
    }
    return seed;
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
    if (!attributesEqualCSE(lhs, rhs)) return false;

    return true;
  }
};

// If the nodes are visited in topological order, one pass is enough.
void EliminateCommonSubexpression(std::shared_ptr<Graph>& graph) {
  // Keep iterating until reach the fixed point.
  bool topological = graph->topological();
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
      auto subit = subexprs.find(node);
      if (subit == subexprs.end()) {
        // If not put current node into the map
        subexprs.insert(node);
      } else {
        // Subexpression exists, replace the uses of node, and destroy it.
        auto existing = *subit;
        JIT_ASSERT(existing != node);
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
        it.destroyCurrent();
        reach_fixed = false;
      }
    }
    if (topological) break;
  }
}

}}
