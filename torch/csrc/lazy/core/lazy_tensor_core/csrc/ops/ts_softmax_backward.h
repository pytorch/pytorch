#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class TSSoftmaxBackward : public Node {
 public:
  TSSoftmaxBackward(const Value& grad_output, const Value& output,
                    lazy_tensors::int64 dim, const Value& self);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 dim() const { return dim_; }

 private:
  // The dimension along which the result is computed.
  lazy_tensors::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
