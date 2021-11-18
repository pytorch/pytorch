#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Repeat : public torch::lazy::TsNode {
 public:
  Repeat(const torch::lazy::Value& input, std::vector<int64_t> repeats);

  std::string ToString() const override;

  const std::vector<int64_t>& repeats() const { return repeats_; }

 private:
  // The number of repeats along each dimension.
  std::vector<int64_t> repeats_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
