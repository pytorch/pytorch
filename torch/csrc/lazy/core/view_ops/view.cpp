#include <torch/csrc/lazy/core/view_ops/view.h>

#include <ATen/InferSize.h>

namespace torch {
namespace lazy {

namespace {

Shape NodeOutputShape(const Value& input, c10::ArrayRef<int64_t> output_sizes) {
  const Shape& input_shape = input.shape();
  const auto complete_output_sizes =
      at::infer_size(output_sizes, input_shape.numel());
  return Shape(input_shape.scalar_type(), complete_output_sizes);
}

} // namespace

View::View(const Value& input, std::vector<int64_t> output_size)
    : TsNode(
          OpKind(at::aten::view),
          {input},
          {NodeOutputShape(input, output_size)},
          /*num_outputs=*/1,
          MHash(output_size)),
      output_size_(std::move(output_size)) {}

std::string View::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", output_size=(" << c10::Join(", ", output_size_)
     << ")";
  return ss.str();
}

} // namespace lazy
} // namespace torch
