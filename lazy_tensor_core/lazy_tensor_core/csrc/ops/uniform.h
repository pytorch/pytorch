#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Uniform : public TsNode {
 public:
  Uniform(const torch::lazy::Value& from, const torch::lazy::Value& to, const torch::lazy::Value& seed,
          const lazy_tensors::Shape& rng_shape);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
