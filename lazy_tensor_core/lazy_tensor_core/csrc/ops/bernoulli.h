#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Bernoulli : public TsNode {
 public:
  Bernoulli(const torch::lazy::Value& probability, const torch::lazy::Value& seed,
            lazy_tensors::Shape shape);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
