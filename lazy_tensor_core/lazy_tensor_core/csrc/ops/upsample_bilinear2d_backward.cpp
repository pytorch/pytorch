#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

UpsampleBilinearBackward::UpsampleBilinearBackward(
    const torch::lazy::Value& input, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size, bool align_corners)
    : TsNode(torch::lazy::OpKind(at::aten::upsample_bilinear2d_backward),
             {input},
             /*num_outputs=*/1,
             torch::lazy::MHash(output_size, input_size, align_corners)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)),
      align_corners_(align_corners) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr UpsampleBilinearBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<UpsampleBilinearBackward>(operands.at(0), output_size_,
                                            input_size_, align_corners_);
}

std::string UpsampleBilinearBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=(" << c10::Join(", ", output_size_)
     << "), input_size=(" << c10::Join(", ", input_size_)
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
