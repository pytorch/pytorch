#include <torch/csrc/lazy/ts_backend/ops/cast.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/tensor_util.h>

namespace torch {
namespace lazy {

namespace {

Shape NodeOutputShape(const Value& input, c10::ScalarType type) {
  Shape shape = input.shape();
  shape.set_scalar_type(type);
  return shape;
}

} // namespace
Cast::Cast(
    const Value& input,
    at::ScalarType dtype,
    c10::optional<at::ScalarType> stype)
    : TsNode(
          ltc_cast,
          {input},
          {NodeOutputShape(input, dtype)},
          /*num_outputs=*/1,
          MHash(101, static_cast<int>(dtype), OptionalOr<int>(stype, -1))),
      dtype_(dtype),
      stype_(stype) {}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString();
  ss << ", dtype=" << dtype_;
  if (stype_) {
    ss << ", stype=" << *stype_;
  }
  return ss.str();
}

} // namespace lazy
} // namespace torch
