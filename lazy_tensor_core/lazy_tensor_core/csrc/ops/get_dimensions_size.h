#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class GetDimensionsSize : public torch::lazy::TsNode {
 public:
  GetDimensionsSize(const torch::lazy::Value& input,
                    std::vector<int64_t> dimensions);

  std::string ToString() const override;

  const std::vector<int64_t>& sizes() const { return dimensions_; }

 private:
  std::vector<int64_t> dimensions_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
