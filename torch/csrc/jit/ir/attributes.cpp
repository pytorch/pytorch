#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

AttributeValue::Ptr GraphAttr::clone() const {
  return Ptr(new GraphAttr(name, value_->copy()));
}

std::unique_ptr<AttributeValue> GraphsAttr::clone() const {
  std::vector<std::shared_ptr<Graph>> copy(value_.size());
  for (const auto i : c10::irange(value_.size())) {
    copy[i] = value_.at(i)->copy();
  }
  return Ptr(new GraphsAttr(name, std::move(copy)));
}

template struct ScalarAttributeValue<c10::complex<double>, AttributeKind::c>;
template struct VectorAttributeValue<c10::complex<double>, AttributeKind::cs>;
template struct ScalarAttributeValue<double, AttributeKind::f>;
template struct VectorAttributeValue<double, AttributeKind::fs>;
template struct ScalarAttributeValue<int64_t, AttributeKind::i>;
template struct VectorAttributeValue<int64_t, AttributeKind::is>;
template struct ScalarAttributeValue<std::string, AttributeKind::s>;
template struct VectorAttributeValue<std::string, AttributeKind::ss>;
template struct ScalarAttributeValue<at::Tensor, AttributeKind::t>;
template struct VectorAttributeValue<at::Tensor, AttributeKind::ts>;
template struct ScalarAttributeValue<c10::TypePtr, AttributeKind::ty>;
template struct VectorAttributeValue<c10::TypePtr, AttributeKind::tys>;
template struct ScalarAttributeValue<at::IValue, AttributeKind::ival>;

} // namespace jit
} // namespace torch
