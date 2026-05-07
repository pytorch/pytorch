#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

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

} // namespace torch::jit
