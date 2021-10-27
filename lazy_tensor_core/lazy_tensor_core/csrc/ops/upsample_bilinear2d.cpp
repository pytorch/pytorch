#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

UpsampleBilinear::UpsampleBilinear(const torch::lazy::Value& input,
                                   std::vector<int64_t> output_size,
                                   bool align_corners)
    : TsNode(torch::lazy::OpKind(at::aten::upsample_bilinear2d), {input},
             /*num_outputs=*/1, torch::lazy::MHash(output_size, align_corners)),
      output_size_(std::move(output_size)),
      align_corners_(align_corners) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr UpsampleBilinear::Clone(OpList operands) const {
  return torch::lazy::MakeNode<UpsampleBilinear>(operands.at(0), output_size_,
                                    align_corners_);
}

std::string UpsampleBilinear::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=(" << c10::Join(", ", output_size_)
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
