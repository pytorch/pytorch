#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ScatterAdd : public TsNode {
 public:
  ScatterAdd(const torch::lazy::Value& input, const torch::lazy::Value& index, const torch::lazy::Value& src,
             lazy_tensors::int64 dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 dim() const { return dim_; };

 private:
  lazy_tensors::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
