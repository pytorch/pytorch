#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ReflectionPad2dBackward : public TsNode {
 public:
  ReflectionPad2dBackward(const torch::lazy::Value& gard_output,
                          const torch::lazy::Value& input,
                          std::vector<int64_t> padding);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& padding() const { return padding_; }

 private:
  std::vector<int64_t> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
