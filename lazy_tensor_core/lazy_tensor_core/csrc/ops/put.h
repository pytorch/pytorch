#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Put : public TsNode {
 public:
  Put(const torch::lazy::Value& input, const torch::lazy::Value& index, const torch::lazy::Value& source,
      bool accumulate);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  bool accumulate() const { return accumulate_; }

 private:
  bool accumulate_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
