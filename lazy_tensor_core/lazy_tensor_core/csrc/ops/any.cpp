#include "lazy_tensor_core/csrc/ops/any.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Any::Any(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> dimensions,
         bool keep_reduced_dimensions)
    : TsNode(torch::lazy::OpKind(at::aten::any), {input},
           /*num_outputs=*/1,
           torch::lazy::MHash(dimensions, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Any::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Any>(operands.at(0), dimensions_, keep_reduced_dimensions_);
}

std::string Any::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dimensions=("
     << lazy_tensors::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
