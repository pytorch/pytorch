
copy: fbcode/caffe2/torch/csrc/jit/ir/attributes.cpp
copyrev: d5d62177f7f9998c9fe8fd55b12dfc3ab48b6691

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
