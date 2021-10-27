#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensors/primitive_types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AdaptiveAvgPool3d : public TsNode {
 public:
  AdaptiveAvgPool3d(const torch::lazy::Value& input,
                    std::vector<int64_t> output_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

 private:
  std::vector<int64_t> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
