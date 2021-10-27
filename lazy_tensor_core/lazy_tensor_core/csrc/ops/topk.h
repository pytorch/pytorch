#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class TopK : public TsNode {
 public:
  TopK(const torch::lazy::Value& input, int64_t k, int64_t dim, bool largest,
       bool sorted);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  int64_t k() const { return k_; };

  int64_t dim() const { return dim_; };

  bool largest() const { return largest_; }

  bool sorted() const { return sorted_; }

 private:
  int64_t k_;
  int64_t dim_;
  bool largest_;
  bool sorted_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
