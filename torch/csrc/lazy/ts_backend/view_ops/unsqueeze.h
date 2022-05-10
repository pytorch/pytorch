#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Unsqueeze : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::unsqueeze);
  }

  Unsqueeze(const torch::lazy::Value& input, int dim);

  std::string ToString() const override;

  int dim() const {
    return dim_;
  }

 private:
  int dim_;
};

} // namespace lazy
} // namespace torch
