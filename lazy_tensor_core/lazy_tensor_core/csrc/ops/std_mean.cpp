#include "lazy_tensor_core/csrc/ops/std_mean.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

StdMean::StdMean(const torch::lazy::Value& input,
                 std::vector<int64_t> dimensions, int64_t correction,
                 bool keep_reduced_dimensions)
    : TsNode(
          torch::lazy::OpKind(at::aten::std_mean), {input},
          /*num_outputs=*/2,
          torch::lazy::MHash(dimensions, correction, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr StdMean::Clone(OpList operands) const {
  return torch::lazy::MakeNode<StdMean>(operands.at(0), dimensions_, correction_,
                           keep_reduced_dimensions_);
}

std::string StdMean::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dimensions=(" << c10::Join(", ", dimensions_)
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
