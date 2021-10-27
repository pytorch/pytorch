#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class UpsampleNearestBackward : public TsNode {
 public:
  UpsampleNearestBackward(const torch::lazy::Value& input,
                          std::vector<int64_t> output_size,
                          std::vector<int64_t> input_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

  const std::vector<int64_t>& input_size() const { return input_size_; }

 private:
  std::vector<int64_t> output_size_;
  std::vector<int64_t> input_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
