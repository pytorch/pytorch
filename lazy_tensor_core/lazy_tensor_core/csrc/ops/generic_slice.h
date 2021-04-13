#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class GenericSlice : public Node {
 public:
  GenericSlice(const Value& input,
               lazy_tensors::Span<const lazy_tensors::int64> base_indices,
               lazy_tensors::Span<const lazy_tensors::int64> sizes);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& base_indices() const {
    return base_indices_;
  }

  const std::vector<lazy_tensors::int64>& sizes() const { return sizes_; }

 private:
  std::vector<lazy_tensors::int64> base_indices_;
  std::vector<lazy_tensors::int64> sizes_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
