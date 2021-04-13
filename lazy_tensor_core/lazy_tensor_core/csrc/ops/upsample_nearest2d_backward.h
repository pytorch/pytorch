#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class UpsampleNearestBackward : public Node {
 public:
  UpsampleNearestBackward(const Value& input,
                          std::vector<lazy_tensors::int64> output_size,
                          std::vector<lazy_tensors::int64> input_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& output_size() const {
    return output_size_;
  }

  const std::vector<lazy_tensors::int64>& input_size() const {
    return input_size_;
  }

 private:
  std::vector<lazy_tensors::int64> output_size_;
  std::vector<lazy_tensors::int64> input_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
