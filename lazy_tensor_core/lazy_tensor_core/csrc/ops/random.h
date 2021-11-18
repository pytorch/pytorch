#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Random : public torch::lazy::TsNode {
 public:
  Random(const torch::lazy::Value& input, const c10::optional<int64_t>& from, const c10::optional<int64_t>& to);

  std::string ToString() const override;
  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override;

  c10::optional<int64_t> from;
  c10::optional<int64_t> to;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
