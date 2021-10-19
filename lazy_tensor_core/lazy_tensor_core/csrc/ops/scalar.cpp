#include "lazy_tensor_core/csrc/ops/scalar.h"

#include <functional>
#include <sstream>

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Scalar::Scalar(const at::Scalar& value, lazy_tensors::Shape shape)
    : TsNode(OpKind(at::prim::Constant), std::move(shape), /*num_outputs=*/1,
           ScalarHash(value)),
      value_(std::move(value)) {}

Scalar::Scalar(const at::Scalar& value, lazy_tensors::PrimitiveType type)
    : TsNode(OpKind(at::prim::Constant),
           lazy_tensors::ShapeUtil::MakeShape(type, {}),
           /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

std::string Scalar::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", value=" << value_;
  return ss.str();
}

NodePtr Scalar::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Scalar>(value_, shape());
}

torch::lazy::hash_t ScalarHash(const at::Scalar& s) {
  return s.isFloatingPoint() ? torch::lazy::Hash(s.toDouble())
                             : torch::lazy::Hash(s.toLong());
}

std::ostream& operator<<(std::ostream& ostrm, at::Scalar s) {
  return ostrm << (s.isFloatingPoint() ? s.toDouble() : s.toLong());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
