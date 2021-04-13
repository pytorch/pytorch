#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class UpsampleNearest : public Node {
 public:
  UpsampleNearest(const Value& input,
                  std::vector<lazy_tensors::int64> output_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& output_size() const {
    return output_size_;
  }

 private:
  std::vector<lazy_tensors::int64> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
