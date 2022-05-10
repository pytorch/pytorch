#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class Normal : public torch::lazy::TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind::Get("aten::normal_");
  }

  Normal(const torch::lazy::Value& self, const double& mean, const double& std, std::vector<torch::lazy::Shape>&& shapes);

  std::string ToString() const override;
  torch::lazy::TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                   torch::lazy::TSLoweringContext* loctx) const override;

  double mean_;
  double std_;
};

}  // namespace lazy
}  // namespace torch
