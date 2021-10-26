#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ReplicationPad : public TsNode {
 public:
  ReplicationPad(const torch::lazy::Value& input, std::vector<int64_t> padding);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& padding() const { return padding_; }

 private:
  std::vector<int64_t> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
