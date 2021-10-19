#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ConvolutionBackwardOverrideable::ConvolutionBackwardOverrideable(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input, const torch::lazy::Value& weight,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding, lazy_tensors::int64 groups)
    : TsNode(torch::lazy::OpKind(at::aten::convolution_backward_overrideable),
           {grad_output, input, weight},
           /*num_outputs=*/3,
           torch::lazy::MHash(stride, padding, dilation, transposed,
                                     output_padding, groups)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      dilation_(std::move(dilation)),
      output_padding_(std::move(output_padding)),
      transposed_(transposed),
      groups_(groups) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

ConvolutionBackwardOverrideable::ConvolutionBackwardOverrideable(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input, const torch::lazy::Value& weight,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding, lazy_tensors::int64 groups,
    std::array<bool, 3> output_mask)
    : ConvolutionBackwardOverrideable(grad_output, input, weight, stride,
                                      padding, dilation, transposed,
                                      output_padding, groups) {
  output_mask_ = std::move(output_mask);
}

NodePtr ConvolutionBackwardOverrideable::Clone(OpList operands) const {
  return torch::lazy::MakeNode<ConvolutionBackwardOverrideable>(
      operands.at(0), operands.at(1), operands.at(2), stride_, padding_,
      dilation_, transposed_, output_padding_, groups_);
}

std::string ConvolutionBackwardOverrideable::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", stride=(" << lazy_tensors::StrJoin(stride_, ", ")
     << "), padding=(" << lazy_tensors::StrJoin(padding_, ", ")
     << "), dilation=(" << lazy_tensors::StrJoin(dilation_, ", ")
     << "), transpose=" << transposed_ << ", output_padding=("
     << lazy_tensors::StrJoin(output_padding_, ", ") << "), groups=" << groups_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
