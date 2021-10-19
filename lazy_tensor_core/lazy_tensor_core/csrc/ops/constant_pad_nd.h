#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ConstantPadNd : public TsNode {
 public:
  ConstantPadNd(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> pad,
                const at::Scalar& value);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const at::Scalar& value() const { return value_; }

  const std::vector<lazy_tensors::int64>& pad() const { return pad_; }

 private:
  std::vector<lazy_tensors::int64> pad_;
  at::Scalar value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
