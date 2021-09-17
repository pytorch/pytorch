#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Amax : public Node {
 public:
  Amax(const Value& input, std::vector<lazy_tensors::int64> dimensions,
       bool keepdim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector<lazy_tensors::int64>& dimensions() const {
    return dimensions_;
  };

  bool keepdim() const { return keepdim_; }

 private:
  std::vector<lazy_tensors::int64> dimensions_;
  bool keepdim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
