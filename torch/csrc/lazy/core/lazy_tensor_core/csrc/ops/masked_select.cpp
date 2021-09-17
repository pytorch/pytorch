#include "lazy_tensor_core/csrc/ops/masked_select.h"

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(const Value& input) {
  const lazy_tensors::Shape& input_shape = input.shape();
  lazy_tensors::int64 input_elements =
      lazy_tensors::ShapeUtil::ElementsIn(input_shape);
  lazy_tensors::PrimitiveType size_type =
      GetShapeDimensionType(/*device=*/nullptr);
  lazy_tensors::Shape result_shape = lazy_tensors::ShapeUtil::MakeShape(
      input_shape.element_type(), {input_elements});
  result_shape.set_dynamic_dimension(0, true);
  return lazy_tensors::ShapeUtil::MakeTupleShape(
      {result_shape, lazy_tensors::ShapeUtil::MakeShape(size_type, {})});
}

}  // namespace

MaskedSelect::MaskedSelect(const Value& input, const Value& mask)
    : Node(ir::OpKind(at::aten::masked_select), {input, mask},
           NodeOutputShape(input),
           /*num_outputs=*/2) {}

NodePtr MaskedSelect::Clone(OpList operands) const {
  return MakeNode<MaskedSelect>(operands.at(0), operands.at(1));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
