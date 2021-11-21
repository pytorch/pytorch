#include "lazy_tensor_core/csrc/view_ops/view.h"

#include <ATen/InferSize.h>
#include <torch/csrc/lazy/core/shape.h>

#include "lazy_tensor_core/csrc/helpers.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

torch::lazy::Shape NodeOutputShape(const torch::lazy::Value& input,
                                   c10::ArrayRef<int64_t> output_sizes) {
  const torch::lazy::Shape& input_shape =
      torch::lazy::GetShapeFromTsValue(input);
  const auto complete_output_sizes =
      at::infer_size(output_sizes, input_shape.numel());
  return torch::lazy::Shape(input_shape.scalar_type(), complete_output_sizes);
}

}  // namespace

View::View(const torch::lazy::Value& input, std::vector<int64_t> output_size)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::view), {input},
                          {NodeOutputShape(input, output_size)},
                          /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

std::string View::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", output_size=("
     << c10::Join(", ", output_size_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
