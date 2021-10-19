#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class IndexSelect : public TsNode {
 public:
  IndexSelect(const torch::lazy::Value& input, lazy_tensors::int64 dim, const torch::lazy::Value& index);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 dim() const { return dim_; };

 private:
  lazy_tensors::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
