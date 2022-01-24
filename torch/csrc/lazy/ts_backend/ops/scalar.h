#pragma once

#include <c10/core/Scalar.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

// Differently from Constant, this is a scalar value broadcasted to a shape.
// Even though a Constant could have been used, for simple scalars broadcasted
// to big shapes, the Constant leads to big literals expanded within the
// computation graph.
class TORCH_API Scalar : public TsNode {
 public:
  Scalar(const at::Scalar& value, Shape shape);
  Scalar(const at::Scalar& value, c10::ScalarType type);

  std::string ToString() const override;

  const at::Scalar& value() const {
    return value_;
  }

 private:
  at::Scalar value_;
};

TORCH_API hash_t ScalarHash(const at::Scalar& s);

} // namespace lazy
} // namespace torch
