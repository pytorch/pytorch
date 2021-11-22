#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Unselect : public torch::lazy::TsNode {
 public:
  Unselect(const torch::lazy::Value& target, const torch::lazy::Value& source,
           int64_t dim, int64_t start, int64_t end, int64_t stride);

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

  int64_t start() const { return start_; }

  int64_t end() const { return end_; }

  int64_t stride() const { return stride_; }

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
