#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class SVD : public torch::lazy::TsNode {
 public:
  SVD(const torch::lazy::Value& input, bool some, bool compute_uv);

  std::string ToString() const override;

  bool some() const { return some_; }

  bool compute_uv() const { return compute_uv_; }

 private:
  bool some_;
  bool compute_uv_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
