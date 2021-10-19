#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class GetDimensionsSize : public TsNode {
 public:
  GetDimensionsSize(const torch::lazy::Value& input,
                    std::vector<lazy_tensors::int64> dimensions);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& dimensions() const {
    return dimensions_;
  }

 private:
  std::vector<lazy_tensors::int64> dimensions_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
