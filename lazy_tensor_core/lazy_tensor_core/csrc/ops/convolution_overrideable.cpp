#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ConvolutionOverrideable::ConvolutionOverrideable(
    const torch::lazy::Value& input, const torch::lazy::Value& weight,
    const torch::lazy::Value& bias, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::convolution_overrideable),
          {input, weight, bias},
          /*num_outputs=*/1,
          torch::lazy::MHash(stride, padding, dilation, transposed,
                             output_padding, groups)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      dilation_(std::move(dilation)),
      output_padding_(std::move(output_padding)),
      transposed_(transposed),
      groups_(groups) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

ConvolutionOverrideable::ConvolutionOverrideable(
    const torch::lazy::Value& input, const torch::lazy::Value& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::convolution_overrideable),
          {input, weight},
          /*num_outputs=*/1,
          torch::lazy::MHash(stride, padding, dilation, transposed,
                             output_padding, groups)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      dilation_(std::move(dilation)),
      output_padding_(std::move(output_padding)),
      transposed_(transposed),
      groups_(groups) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string ConvolutionOverrideable::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", stride=("
     << c10::Join(", ", stride_) << "), padding=(" << c10::Join(", ", padding_)
     << "), dilation=(" << c10::Join(", ", dilation_)
     << "), transpose=" << transposed_ << ", output_padding=("
     << c10::Join(", ", output_padding_) << "), groups=" << groups_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
