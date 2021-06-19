#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Expand : public Node {
 public:
  Expand(const Value& input, std::vector<lazy_tensors::int64> size,
         bool is_scalar_expand);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<lazy_tensors::int64>& size() const { return size_; };

  bool is_scalar_expand() const { return is_scalar_expand_; }

 private:
  std::vector<lazy_tensors::int64> size_;
  // True iff the input was a scalar and this was generated internally by a
  // lowering and not by user action. For some backends, this difference can be
  // material (for example setting strides according to eager semantics).
  bool is_scalar_expand_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
