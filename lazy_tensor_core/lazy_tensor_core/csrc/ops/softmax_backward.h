#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class SoftmaxBackward : public TsNode {
 public:
  SoftmaxBackward(const torch::lazy::Value& grad_output,
                  const torch::lazy::Value& output, int64_t dim);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

 private:
  // The dimension along which the result is computed.
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
