#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Normal : public TsNode {
 public:
  Normal(const torch::lazy::Value& mean, const torch::lazy::Value& std, const torch::lazy::Value& seed);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
