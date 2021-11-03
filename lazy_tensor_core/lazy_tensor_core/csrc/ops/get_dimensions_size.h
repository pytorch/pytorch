#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class GetDimensionsSize : public TsNode {
 public:
  GetDimensionsSize(const torch::lazy::Value& input,
                    std::vector<int64_t> dimensions);

  std::string ToString() const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

 private:
  std::vector<int64_t> dimensions_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
