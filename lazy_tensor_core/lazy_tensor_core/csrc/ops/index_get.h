#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class IndexGet : public Node {
 public:
  IndexGet(const ir::Value& base, const ir::Value& indices,
           lazy_tensors::int64 start_dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 start_dim() const { return start_dim_; }

 private:
  // The dimension number at which indexing starts.
  lazy_tensors::int64 start_dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
