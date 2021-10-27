#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Amax : public TsNode {
 public:
  Amax(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
       bool keepdim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; };

  bool keepdim() const { return keepdim_; }

 private:
  std::vector<int64_t> dimensions_;
  bool keepdim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
