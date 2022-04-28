#include <torch/csrc/lazy/core/view_ops/unsqueeze.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch {
namespace lazy {

std::vector<int64_t> BuildUnsqueezedDimensions(
    c10::ArrayRef<int64_t> dimensions,
    int64_t squeeze_dim) {
  std::vector<int64_t> output_dimensions(
      dimensions.cbegin(), dimensions.cend());
  output_dimensions.insert(output_dimensions.begin() + squeeze_dim, 1);
  return output_dimensions;
}

Unsqueeze::Unsqueeze(const torch::lazy::Value& input, int dim)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::unsqueeze),
          {input},
          /*num_outputs=*/1,
          torch::lazy::MHash(dim)),
      dim_(dim) {
  addComputedShape([&]() {
    const auto& input_shape = input.shape();
    return torch::lazy::Shape(
        input_shape.scalar_type(),
        BuildUnsqueezedDimensions(input_shape.sizes(), dim));
  });
}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

} // namespace lazy
} // namespace torch
