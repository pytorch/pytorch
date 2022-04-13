#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

std::vector<int64_t> BuildUnsqueezeDimensions(c10::ArrayRef<int64_t> dimensions,
                                              int64_t dim);

class Unsqueeze : public torch::lazy::TsNode {
 public:
  // Insert a dimension of size one at the specified position.
  Unsqueeze(const torch::lazy::Value& input, int dim);

  bool Equal(const torch::lazy::Value& input, int dim) const {
    size_t i = 0;
    return (operand(i++) == input && dim_ == dim);
  }

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  // Position to unsqueeze.
  int dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
