#include "torch/csrc/jit/ir.h"

#include <algorithm>
#include <unordered_map>

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

namespace torch { namespace jit {



// Check whether two nodes have the same attributes in CSE.
// This function may be too conservative for general use.
// Do NOT support t/ts/g/gs attributes.
// If t/ts are supported, CONSTANT node comparison may need to consider device.
bool attributesEqualCSE(const Node* lhs, const Node* rhs) {
  JIT_ASSERT(lhs != nullptr);
  JIT_ASSERT(rhs != nullptr);
  // One has attributes, the other does not.
  if (lhs->hasAttributes() != rhs->hasAttributes()) return false;
  // Neither has attributes.
  if (!lhs->hasAttributes() && !rhs->hasAttributes()) return true;

  auto lnames = lhs->attributeNames();
  auto rnames = rhs->attributeNames();
  if (lnames != rnames) return false;

  for (auto name : lnames) {
    if (lhs->kindOf(name) != rhs->kindOf(name)) return false;

    #define COMPARE_ATTRIBUTEVALUE(type) \
      case AttributeKind::type: \
        { if (lhs->type(name) != rhs->type(name)) return false; } break;

    switch(lhs->kindOf(name)) {
      COMPARE_ATTRIBUTEVALUE(f)
      COMPARE_ATTRIBUTEVALUE(fs)
      COMPARE_ATTRIBUTEVALUE(i)
      COMPARE_ATTRIBUTEVALUE(is)
      COMPARE_ATTRIBUTEVALUE(s)
      COMPARE_ATTRIBUTEVALUE(ss)
      default:
        // NB: Comparison of nodes with tensor(s) or graph(s) will return false.
        return false;
    }

    #undef COMPARE_ATTRIBUTEVALUE
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

    // Check whether the inputs are the same.
    if (lhs->inputs().size() != rhs->inputs().size()) return false;

    if (!std::equal(lhs->inputs().begin(), lhs->inputs().end(), rhs->inputs().begin())) return false;

    // Check the attributes.
    if (!attributesEqualCSE(lhs, rhs)) return false;

    return true;
  }
};

// The function implements common subexpression elimination.
// Since the nodes are visited in topological order, one pass is enough.
void EliminateCommonSubexpression(std::shared_ptr<Graph>& graph) {
  auto nodes = graph->nodes();
  std::unordered_set<Node*, HashNodeCSE, EqualNodeCSE> subexprs;
  for (auto it = nodes.begin(); it != nodes.end(); ++ it) {
    auto node = *it;
    if (node->kind() == kPythonOp
        || node->kind() == kCppOp
        || node->kind() == kEval
       ) {
      // Do NOT have enough information to do CSE on these nodes.
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
    }
  }
}

}}
