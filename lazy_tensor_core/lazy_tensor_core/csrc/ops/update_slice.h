#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class UpdateSlice : public Node {
 public:
  UpdateSlice(const Value& input, const Value& source,
              lazy_tensors::Span<const lazy_tensors::int64> base_indices);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& base_indices() const {
    return base_indices_;
  }

 private:
  std::vector<lazy_tensors::int64> base_indices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
