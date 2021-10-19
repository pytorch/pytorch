#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class IndexPut : public TsNode {
 public:
  IndexPut(const torch::lazy::Value& base, const torch::lazy::Value& indices,
           lazy_tensors::int64 start_dim, const torch::lazy::Value& values,
           bool accumulate);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 start_dim() const { return start_dim_; }

  bool accumulate() const { return accumulate_; }

 private:
  // The dimension number at which indexing starts.
  lazy_tensors::int64 start_dim_;
  // Whether to accumulate instead of set.
  bool accumulate_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
