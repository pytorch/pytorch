#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ReflectionPad2d : public TsNode {
 public:
  ReflectionPad2d(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> padding);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& padding() const { return padding_; }

 private:
  std::vector<lazy_tensors::int64> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
