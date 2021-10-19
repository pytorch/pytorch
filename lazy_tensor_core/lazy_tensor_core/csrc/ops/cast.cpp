#include "lazy_tensor_core/csrc/ops/cast.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/primitive_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const torch::lazy::Value& input,
                                    lazy_tensors::PrimitiveType type) {
  lazy_tensors::Shape shape = ir::GetShapeFromTsValue(input);
  shape.set_element_type(type);
  return shape;
}

}  // namespace

Cast::Cast(const torch::lazy::Value& input, lazy_tensors::PrimitiveType type)
    : TsNode(ltc_cast, {input}, NodeOutputShape(input, type),
           /*num_outputs=*/1,
           torch::lazy::MHash(static_cast<int>(type))),
      type_(type) {}

Cast::Cast(const torch::lazy::Value& input, at::ScalarType dtype,
           c10::optional<at::ScalarType> stype)
    : TsNode(ltc_cast, {input},
           NodeOutputShape(input,
                           MakeLtcPrimitiveType(dtype, /*device=*/nullptr)),
           /*num_outputs=*/1,
           torch::lazy::MHash(101, static_cast<int>(dtype),
                                     OptionalOr<int>(stype, -1))),
      type_(MakeLtcPrimitiveType(dtype, /*device=*/nullptr)),
      dtype_(dtype),
      stype_(stype) {}

NodePtr Cast::Clone(OpList operands) const {
  return dtype_ ? torch::lazy::MakeNode<Cast>(operands.at(0), *dtype_, stype_)
                : torch::lazy::MakeNode<Cast>(operands.at(0), type_);
}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", type="
     << lazy_tensors::primitive_util::LowercasePrimitiveTypeName(type_);
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  if (stype_) {
    ss << ", stype=" << *stype_;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
