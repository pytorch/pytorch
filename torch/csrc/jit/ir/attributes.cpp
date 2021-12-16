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

template class ScalarAttributeValue<c10::complex<double>, AttributeKind::c>;
template class VectorAttributeValue<c10::complex<double>, AttributeKind::cs>;
template class ScalarAttributeValue<double, AttributeKind::f>;
template class VectorAttributeValue<double, AttributeKind::fs>;
template class ScalarAttributeValue<int64_t, AttributeKind::i>;
template class VectorAttributeValue<int64_t, AttributeKind::is>;
template class ScalarAttributeValue<std::string, AttributeKind::s>;
template class VectorAttributeValue<std::string, AttributeKind::ss>;
template class ScalarAttributeValue<at::Tensor, AttributeKind::t>;
template class VectorAttributeValue<at::Tensor, AttributeKind::ts>;
template class ScalarAttributeValue<c10::TypePtr, AttributeKind::ty>;
template class VectorAttributeValue<c10::TypePtr, AttributeKind::tys>;
template class ScalarAttributeValue<at::IValue, AttributeKind::ival>;

} // namespace jit
} // namespace torch
