#pragma once

#include <c10/core/Scalar.h>

#include <iostream>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/computation_client/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the
// computation graph.
class Scalar : public Node {
 public:
  Scalar(const at::Scalar& value, lazy_tensors::Shape shape);
  Scalar(const at::Scalar& value, lazy_tensors::PrimitiveType type);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const at::Scalar& value() const { return value_; }

 private:
  at::Scalar value_;
};

lazy_tensors::hash_t ScalarHash(const at::Scalar& s);

std::ostream& operator<<(std::ostream& ostrm, at::Scalar s);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
