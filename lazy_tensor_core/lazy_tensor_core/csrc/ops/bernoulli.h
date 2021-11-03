#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Bernoulli : public TsNode {
 public:
  Bernoulli(const torch::lazy::Value& probability, const torch::lazy::Value& seed,
            lazy_tensors::Shape shape);
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
