#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Resize : public TsNode {
 public:
  Resize(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& size() const { return size_; }

 private:
  std::vector<lazy_tensors::int64> size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
