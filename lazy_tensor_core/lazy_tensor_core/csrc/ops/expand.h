#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Expand : public Node {
 public:
  Expand(const Value& input, std::vector<lazy_tensors::int64> size);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<lazy_tensors::int64>& size() const { return size_; };

 private:
  std::vector<lazy_tensors::int64> size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
