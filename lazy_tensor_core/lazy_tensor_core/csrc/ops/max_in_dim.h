#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MaxInDim : public TsNode {
 public:
  MaxInDim(const torch::lazy::Value& input, int64_t dim, bool keepdim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  int64_t dim() const { return dim_; };

  bool keepdim() const { return keepdim_; }

 private:
  int64_t dim_;
  bool keepdim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
