#include "lazy_tensor_core/csrc/ops/cast.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const torch::lazy::Value& input,
                                    c10::ScalarType type) {
  lazy_tensors::Shape shape = ir::GetShapeFromTsValue(input);
  shape.set_element_type(type);
  return shape;
}

}  // namespace
Cast::Cast(const torch::lazy::Value& input, at::ScalarType dtype,
           c10::optional<at::ScalarType> stype)
    : TsNode(ltc_cast, {input}, NodeOutputShape(input, dtype),
             /*num_outputs=*/1,
             torch::lazy::MHash(101, static_cast<int>(dtype),
                                OptionalOr<int>(stype, -1))),
      dtype_(dtype),
      stype_(stype) {}

NodePtr Cast::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Cast>(operands.at(0), dtype_, stype_);
}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString();
  ss << ", dtype=" << dtype_;
  if (stype_) {
    ss << ", stype=" << *stype_;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
