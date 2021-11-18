#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Bernoulli : public torch::lazy::TsNode {
 public:
  Bernoulli(const torch::lazy::Value& probability, const torch::lazy::Value& seed,
            torch::lazy::Shape shape);
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
