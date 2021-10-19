#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MaxUnpoolNdBackward : public TsNode {
 public:
  MaxUnpoolNdBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                      const torch::lazy::Value& indices,
                      std::vector<lazy_tensors::int64> output_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& output_size() const {
    return output_size_;
  }

 private:
  std::vector<lazy_tensors::int64> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
