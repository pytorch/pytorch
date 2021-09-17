#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Permute : public Node {
 public:
  Permute(const Value& input, std::vector<lazy_tensors::int64> dims);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& dims() const { return dims_; }

  static lazy_tensors::Shape MakePermuteShape(
      const lazy_tensors::Shape& source_shape,
      lazy_tensors::Span<const lazy_tensors::int64> permutation);

 private:
  // The permutation of dimensions.
  std::vector<lazy_tensors::int64> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
