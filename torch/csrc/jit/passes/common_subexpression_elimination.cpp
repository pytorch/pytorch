#include "torch/csrc/jit/ir.h"

#include <algorithm>
#include <unordered_map>

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/hash.h"

namespace torch { namespace jit {

namespace {

bool tensorEqual(const at::Tensor& lhs, const at::Tensor& rhs) {
  return &lhs.type() == &rhs.type() && lhs.equal(rhs);
}

bool tensorListEqual(const std::vector<at::Tensor>& lhs, const std::vector<at::Tensor>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), tensorEqual);
}


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
  std::sort(lnames.begin(), lnames.end());
  std::sort(rnames.begin(), rnames.end());
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
      case AttributeKind::t: {
        if (!tensorEqual(lhs->t(name), rhs->t(name))) return false;
        break;
      }
      case AttributeKind::ts: {
        if (!tensorListEqual(lhs->ts(name), rhs->ts(name))) return false;
        break;
      }
      case AttributeKind::g:
      case AttributeKind::gs:
        return false;
    }

    #undef COMPARE_ATTRIBUTEVALUE
  }

  return true;
}

struct HashNodeCSE {
  size_t operator()(const Node* k) const {
    JIT_ASSERT(k != nullptr);
    return get_hash(k->kind(),
                    k->stage(),
                    fmap(k->inputs(), [](const Value *v) { return v->unique(); }));
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
    auto lhs_inputs = lhs->inputs();
    auto rhs_inputs = rhs->inputs();

    if (lhs_inputs.size() != rhs_inputs.size()) return false;

    if (!std::equal(lhs_inputs.begin(), lhs_inputs.end(), rhs_inputs.begin())) return false;

    // Check the attributes.
    if (!attributesEqualCSE(lhs, rhs)) return false;

    return true;
  }
};

} // anonymous namespace

// The function implements common subexpression elimination.
// Since the nodes are visited in topological order, one pass is enough.
void EliminateCommonSubexpression(Block * block) {
  std::unordered_set<Node*, HashNodeCSE, EqualNodeCSE> subexprs;
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++ it) {
    auto node = *it;
    if (node->kind() == prim::PythonOp
        || node->kind() == prim::CppOp
        || node->kind() == prim::Eval
        || node->blocks().size() > 0
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
      node->replaceAllUsesWith(existing);
      // Destroy the node.
      it.destroyCurrent();
    }
  }
}

void EliminateCommonSubexpression(std::shared_ptr<Graph>& graph) {
  EliminateCommonSubexpression(graph->block());
}

}}
