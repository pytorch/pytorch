#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Split the tensor into chunks along a given dimension.
class Split : public Node {
 public:
  Split(const Value& input, std::vector<lazy_tensors::int64> split_sizes,
        lazy_tensors::int64 dim);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& split_sizes() const {
    return split_sizes_;
  }

  lazy_tensors::int64 dim() const { return dim_; }

 private:
  std::vector<lazy_tensors::int64> split_sizes_;
  lazy_tensors::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
