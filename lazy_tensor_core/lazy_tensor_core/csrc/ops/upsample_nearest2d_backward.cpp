#include "lazy_tensor_core/csrc/ops/upsample_nearest2d_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

UpsampleNearestBackward::UpsampleNearestBackward(
    const torch::lazy::Value& input, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size)
    : TsNode(torch::lazy::OpKind(at::aten::upsample_nearest2d_backward),
             {input},
             /*num_outputs=*/1, torch::lazy::MHash(output_size, input_size)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr UpsampleNearestBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<UpsampleNearestBackward>(operands.at(0), output_size_,
                                           input_size_);
}

std::string UpsampleNearestBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=(" << c10::Join(", ", output_size_)
     << "), input_size=(" << c10::Join(", ", input_size_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
