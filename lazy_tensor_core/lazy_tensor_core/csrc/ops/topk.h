#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class TopK : public TsNode {
 public:
  TopK(const torch::lazy::Value& input, lazy_tensors::int64 k, lazy_tensors::int64 dim,
       bool largest, bool sorted);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 k() const { return k_; };

  lazy_tensors::int64 dim() const { return dim_; };

  bool largest() const { return largest_; }

  bool sorted() const { return sorted_; }

 private:
  lazy_tensors::int64 k_;
  lazy_tensors::int64 dim_;
  bool largest_;
  bool sorted_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
