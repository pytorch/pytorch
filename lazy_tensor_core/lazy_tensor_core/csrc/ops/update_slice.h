#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class UpdateSlice : public torch::lazy::TsNode {
 public:
  UpdateSlice(const torch::lazy::Value& input, const torch::lazy::Value& source,
              c10::ArrayRef<int64_t> base_indices);

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const { return base_indices_; }

 private:
  std::vector<int64_t> base_indices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
