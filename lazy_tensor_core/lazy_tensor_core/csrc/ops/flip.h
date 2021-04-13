#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Flip : public Node {
 public:
  Flip(const Value& input, std::vector<lazy_tensors::int64> dims);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& dims() const { return dims_; }

 private:
  // The dimensions which are flipped.
  std::vector<lazy_tensors::int64> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
