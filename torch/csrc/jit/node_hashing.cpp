#include <torch/csrc/jit/ir.h>

#include <algorithm>
#include <unordered_map>

#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/utils/hash.h>

namespace torch {
namespace jit {

namespace {

bool tensorEqual(const at::Tensor& lhs, const at::Tensor& rhs) {
  return lhs.type() == rhs.type() && lhs.equal(rhs);
}

bool tensorListEqual(
    const std::vector<at::Tensor>& lhs,
    const std::vector<at::Tensor>& rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return std::equal(lhs.begin(), lhs.end(), rhs.begin(), tensorEqual);
}

bool typeListEqual(
    const std::vector<TypePtr>& lhs,
    const std::vector<TypePtr>& rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (*lhs[i] != *rhs[i]) {
      return false;
    }
  }
  return true;
}

// Check whether two nodes have the same attributes in CSE.
// This function may be too conservative for general use.
// Do NOT support g/gs attributes.
bool attributesEqualCSE(const Node* lhs, const Node* rhs) {
  AT_ASSERT(lhs != nullptr);
  AT_ASSERT(rhs != nullptr);
  // One has attributes, the other does not.
  if (lhs->hasAttributes() != rhs->hasAttributes())
    return false;
  // Neither has attributes.
  if (!lhs->hasAttributes() && !rhs->hasAttributes())
    return true;

  auto lnames = lhs->attributeNames();
  auto rnames = rhs->attributeNames();
  std::sort(lnames.begin(), lnames.end());
  std::sort(rnames.begin(), rnames.end());
  if (lnames != rnames)
    return false;

  for (auto name : lnames) {
    if (lhs->kindOf(name) != rhs->kindOf(name))
      return false;

#define COMPARE_ATTRIBUTEVALUE(selector)            \
  case AttributeKind::selector: {                   \
    if (lhs->selector(name) != rhs->selector(name)) \
      return false;                                 \
  } break;

    switch (lhs->kindOf(name)) {
      COMPARE_ATTRIBUTEVALUE(f)
      COMPARE_ATTRIBUTEVALUE(fs)
      COMPARE_ATTRIBUTEVALUE(i)
      COMPARE_ATTRIBUTEVALUE(is)
      COMPARE_ATTRIBUTEVALUE(s)
      COMPARE_ATTRIBUTEVALUE(ss)
      case AttributeKind::t: {
        if (!tensorEqual(lhs->t(name), rhs->t(name)))
          return false;
        break;
      }
      case AttributeKind::ts: {
        if (!tensorListEqual(lhs->ts(name), rhs->ts(name)))
          return false;
        break;
      }
      case AttributeKind::ty:
        if (*lhs->ty(name) != *rhs->ty(name)) {
          return false;
        }
      case AttributeKind::tys:
        if (!typeListEqual(lhs->tys(name), rhs->tys(name))) {
          return false;
        }
      case AttributeKind::g:
      case AttributeKind::gs:
        return false;
    }

#undef COMPARE_ATTRIBUTEVALUE
  }

  return true;
}

} // anonymous namespace

size_t HashNode::operator()(const Node* k) const {
  AT_ASSERT(k != nullptr);
  return get_hash(
      k->kind(),
      fmap(k->outputs(), [](const Value* v) { return v->type()->kind(); }),
      fmap(k->inputs(), [](const Value* v) { return v->unique(); }));
};

bool EqualNode::operator()(const Node* lhs, const Node* rhs) const {
  if (lhs == nullptr && rhs == nullptr)
    return true;
  if (lhs == nullptr || rhs == nullptr)
    return false;

  if (lhs->kind() != rhs->kind())
    return false;

  // Check whether the output types are the same.
  auto lhs_outputs = lhs->outputs();
  auto rhs_outputs = rhs->outputs();
  if (lhs_outputs.size() != rhs_outputs.size())
    return false;
  for (size_t i = 0; i < lhs_outputs.size(); ++i) {
    if (*lhs_outputs[i]->type() != *rhs_outputs[i]->type())
      return false;
    if (lhs_outputs[i]->type() == CapsuleType::get())
      return false;
  }

  // Check whether the inputs are the same.
  auto lhs_inputs = lhs->inputs();
  auto rhs_inputs = rhs->inputs();
  if (lhs_inputs.size() != rhs_inputs.size())
    return false;
  if (!std::equal(lhs_inputs.begin(), lhs_inputs.end(), rhs_inputs.begin()))
    return false;

  if (!attributesEqualCSE(lhs, rhs))
    return false;

  return true;
};

} // namespace jit
} // namespace torch
