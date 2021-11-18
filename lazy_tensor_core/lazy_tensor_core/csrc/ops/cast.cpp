#include "lazy_tensor_core/csrc/ops/cast.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

torch::lazy::Shape NodeOutputShape(const torch::lazy::Value& input,
                                    c10::ScalarType type) {
  torch::lazy::Shape shape = torch::lazy::GetShapeFromTsValue(input);
  shape.set_scalar_type(type);
  return shape;
}

}  // namespace
Cast::Cast(const torch::lazy::Value& input, at::ScalarType dtype,
           c10::optional<at::ScalarType> stype)
    : torch::lazy::TsNode(ltc_cast, {input}, {NodeOutputShape(input, dtype)},
                          /*num_outputs=*/1,
                          torch::lazy::MHash(101, static_cast<int>(dtype),
                                             OptionalOr<int>(stype, -1))),
      dtype_(dtype),
      stype_(stype) {}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString();
  ss << ", dtype=" << dtype_;
  if (stype_) {
    ss << ", stype=" << *stype_;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
