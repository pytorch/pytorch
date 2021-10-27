#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Split the tensor into chunks along a given dimension.
class Split : public TsNode {
 public:
  Split(const torch::lazy::Value& input, std::vector<int64_t> split_sizes,
        int64_t dim);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& split_sizes() const { return split_sizes_; }

  int64_t dim() const { return dim_; }

 private:
  std::vector<int64_t> split_sizes_;
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
