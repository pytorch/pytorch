#include "lazy_tensor_core/csrc/ops/unsqueeze.h"

#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const torch::lazy::Value& input, int dim) {
  const lazy_tensors::Shape& shape = ir::GetShapeFromTsValue(input);
  auto dimensions = BuildUnsqueezeDimensions(shape.dimensions(), dim);
  return lazy_tensors::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

}  // namespace

Unsqueeze::Unsqueeze(const torch::lazy::Value& input, int dim)
    : TsNode(torch::lazy::OpKind(at::aten::unsqueeze), {input},
           [&]() { return NodeOutputShape(input, dim); },
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

NodePtr Unsqueeze::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Unsqueeze>(operands.at(0), dim_);
}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
