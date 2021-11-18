#include "lazy_tensor_core/csrc/ops/scalar.h"

#include <functional>
#include <sstream>

#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Scalar::Scalar(const at::Scalar& value, torch::lazy::Shape shape)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::prim::Constant),
                          std::move(shape), /*num_outputs=*/1,
                          ScalarHash(value)),
      value_(std::move(value)) {}

Scalar::Scalar(const at::Scalar& value, c10::ScalarType type)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::prim::Constant),
                          {lazy_tensors::ShapeUtil::MakeShape(type, {})},
                          /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

std::string Scalar::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", value=" << value_;
  return ss.str();
}

torch::lazy::hash_t ScalarHash(const at::Scalar& s) {
  return s.isFloatingPoint() ? torch::lazy::Hash(s.toDouble())
                             : torch::lazy::Hash(s.toLong());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
