#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class UpsampleBilinear : public Node {
 public:
  UpsampleBilinear(const Value& input,
                   std::vector<lazy_tensors::int64> output_size,
                   bool align_corners);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& output_size() const {
    return output_size_;
  }

  bool align_corners() const { return align_corners_; }

 private:
  std::vector<lazy_tensors::int64> output_size_;
  bool align_corners_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
