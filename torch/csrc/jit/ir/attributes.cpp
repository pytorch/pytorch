#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

AttributeValue::Ptr GraphAttr::clone() const {
  return Ptr(new GraphAttr(name, value_->copy()));
}

std::unique_ptr<AttributeValue> GraphsAttr::clone() const {
  std::vector<std::shared_ptr<Graph>> copy(value_.size());
  for (size_t i = 0; i < value_.size(); ++i) {
    copy[i] = value_.at(i)->copy();
  }
  return Ptr(new GraphsAttr(name, std::move(copy)));
}

} // namespace jit
} // namespace torch
