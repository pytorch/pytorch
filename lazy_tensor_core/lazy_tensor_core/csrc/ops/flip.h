#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Flip : public TsNode {
 public:
  Flip(const torch::lazy::Value& input, std::vector<int64_t> dims);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& dims() const { return dims_; }

 private:
  // The dimensions which are flipped.
  std::vector<int64_t> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
