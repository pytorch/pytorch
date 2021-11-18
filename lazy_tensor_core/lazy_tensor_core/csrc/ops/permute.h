#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Permute : public torch::lazy::TsNode {
 public:
  Permute(const torch::lazy::Value& input, std::vector<int64_t> dims);

  std::string ToString() const override;

  const std::vector<int64_t>& dims() const { return dims_; }

  static torch::lazy::Shape MakePermuteShape(
      const torch::lazy::Shape& source_shape,
      c10::ArrayRef<int64_t> permutation);

 private:
  // The permutation of dimensions.
  std::vector<int64_t> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
