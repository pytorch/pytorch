#include "lazy_tensor_core/csrc/ops/std.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Std::Std(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
         bool keep_reduced_dimensions, int64_t correction)
    : TsNode(
          torch::lazy::OpKind(at::aten::std), {input},
          /*num_outputs=*/1,
          torch::lazy::MHash(dimensions, keep_reduced_dimensions, correction)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      correction_(correction) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Std::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Std>(operands.at(0), dimensions_, keep_reduced_dimensions_,
                       correction_);
}

std::string Std::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dimensions=(" << c10::Join(", ", dimensions_)
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
