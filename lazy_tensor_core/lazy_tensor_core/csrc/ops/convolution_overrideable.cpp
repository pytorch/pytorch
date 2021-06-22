#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ConvolutionOverrideable::ConvolutionOverrideable(
    const Value& input, const Value& weight, const Value& bias,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding, lazy_tensors::int64 groups)
    : Node(ir::OpKind(at::aten::convolution_overrideable),
           {input, weight, bias},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(stride, padding, dilation, transposed,
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

ConvolutionOverrideable::ConvolutionOverrideable(
    const Value& input, const Value& weight,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding, lazy_tensors::int64 groups)
    : Node(ir::OpKind(at::aten::convolution_overrideable), {input, weight},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(stride, padding, dilation, transposed,
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

NodePtr ConvolutionOverrideable::Clone(OpList operands) const {
  return operands.size() == 3
             ? MakeNode<ConvolutionOverrideable>(
                   operands.at(0), operands.at(1), operands.at(2), stride_,
                   padding_, dilation_, transposed_, output_padding_, groups_)
             : MakeNode<ConvolutionOverrideable>(
                   operands.at(0), operands.at(1), stride_, padding_, dilation_,
                   transposed_, output_padding_, groups_);
}

std::string ConvolutionOverrideable::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", stride=(" << lazy_tensors::StrJoin(stride_, ", ")
     << "), padding=(" << lazy_tensors::StrJoin(padding_, ", ")
     << "), dilation=(" << lazy_tensors::StrJoin(dilation_, ", ")
     << "), transpose=" << transposed_ << ", output_padding=("
     << lazy_tensors::StrJoin(output_padding_, ", ") << "), groups=" << groups_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
