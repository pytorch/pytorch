#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

static torch::lazy::Shape inferBiasShape(const torch::lazy::Value& grad_output) {
  auto grad_shape = ir::GetShapeFromTsValue(grad_output);
  auto bias_dim = grad_shape.size(1);
  return torch::lazy::Shape(grad_shape.scalar_type(), {bias_dim});
}

ConvolutionBackwardOverrideable::ConvolutionBackwardOverrideable(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    const torch::lazy::Value& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups)
    : TsNode(torch::lazy::OpKind(at::aten::convolution_backward_overrideable),
             {grad_output, input, weight},
             {ir::GetShapeFromTsValue(input), ir::GetShapeFromTsValue(weight),
              inferBiasShape(grad_output)},
             /*num_outputs=*/3,
             torch::lazy::MHash(stride, padding, dilation, transposed,
                                output_padding, groups)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      dilation_(std::move(dilation)),
      output_padding_(std::move(output_padding)),
      transposed_(transposed),
      groups_(groups) {}

ConvolutionBackwardOverrideable::ConvolutionBackwardOverrideable(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
    const torch::lazy::Value& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups,
    std::array<bool, 3> output_mask)
    : ConvolutionBackwardOverrideable(grad_output, input, weight, stride,
                                      padding, dilation, transposed,
                                      output_padding, groups) {
  output_mask_ = std::move(output_mask);
}

std::string ConvolutionBackwardOverrideable::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", stride=(" << c10::Join(", ", stride_)
     << "), padding=(" << c10::Join(", ", padding_) << "), dilation=("
     << c10::Join(", ", dilation_) << "), transpose=" << transposed_
     << ", output_padding=(" << c10::Join(", ", output_padding_)
     << "), groups=" << groups_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
