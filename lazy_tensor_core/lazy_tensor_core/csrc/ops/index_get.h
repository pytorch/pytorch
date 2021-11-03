#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class IndexGet : public TsNode {
 public:
  IndexGet(const torch::lazy::Value& base, const torch::lazy::Value& indices,
           int64_t start_dim);

  std::string ToString() const override;

  int64_t start_dim() const { return start_dim_; }

 private:
  // The dimension number at which indexing starts.
  int64_t start_dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
