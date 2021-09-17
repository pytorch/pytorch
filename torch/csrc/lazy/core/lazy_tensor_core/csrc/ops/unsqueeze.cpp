#include "lazy_tensor_core/csrc/ops/unsqueeze.h"

#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const Value& input, int dim) {
  const lazy_tensors::Shape& shape = input.shape();
  auto dimensions = BuildUnsqueezeDimensions(shape.dimensions(), dim);
  return lazy_tensors::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

}  // namespace

Unsqueeze::Unsqueeze(const Value& input, int dim)
    : Node(ir::OpKind(at::aten::unsqueeze), {input},
           [&]() { return NodeOutputShape(input, dim); },
           /*num_outputs=*/1, lazy_tensors::util::MHash(dim)),
      dim_(dim) {}

NodePtr Unsqueeze::Clone(OpList operands) const {
  return MakeNode<Unsqueeze>(operands.at(0), dim_);
}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
