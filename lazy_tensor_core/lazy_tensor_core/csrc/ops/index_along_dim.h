#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class IndexAlongDim : public TsNode {
 public:
  IndexAlongDim(OpKind op, const torch::lazy::Value& buffer,
                const torch::lazy::Value& index,
                const torch::lazy::Value& value, int64_t dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  int64_t dim() const { return dim_; }

 private:
  // The dimension along which indexing is applied.
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
