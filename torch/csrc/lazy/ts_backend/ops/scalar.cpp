#include <torch/csrc/lazy/ts_backend/ops/scalar.h>

#include <functional>
#include <sstream>

#include <ATen/core/Formatting.h>

namespace torch {
namespace lazy {

using at::operator<<;

Scalar::Scalar(const at::Scalar& value, Shape shape)
    : TsNode(
          OpKind(at::prim::Constant),
          std::move(shape),
          /*num_outputs=*/1,
          ScalarHash(value)),
      value_(value) {}

Scalar::Scalar(const at::Scalar& value, c10::ScalarType type)
    : TsNode(
          OpKind(at::prim::Constant),
          {Shape(type, {})},
          /*num_outputs=*/1,
          ScalarHash(value)),
      value_(value) {}

std::string Scalar::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", value=" << value_;
  return ss.str();
}

hash_t ScalarHash(const at::Scalar& s) {
  return s.isFloatingPoint() ? Hash(s.toDouble()) : Hash(s.toLong());
}

} // namespace lazy
} // namespace torch
