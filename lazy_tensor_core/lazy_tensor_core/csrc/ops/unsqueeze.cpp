#include "lazy_tensor_core/csrc/ops/unsqueeze.h"

#include <torch/csrc/lazy/core/util.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

std::vector<int64_t> BuildUnsqueezeDimensions(c10::ArrayRef<int64_t> dimensions,
                                              int64_t dim) {
  CHECK_LE(dim, dimensions.size());
  std::vector<int64_t> unsqueeze_dimensions(dimensions.begin(), dimensions.end());
  unsqueeze_dimensions.insert(unsqueeze_dimensions.begin() + dim, 1);
  return unsqueeze_dimensions;
}

namespace {

torch::lazy::Shape NodeOutputShape(const torch::lazy::Value& input, int dim) {
  const torch::lazy::Shape& shape = input.shape();
  auto dimensions = BuildUnsqueezeDimensions(shape.sizes(), dim);
  return torch::lazy::Shape(shape.scalar_type(), dimensions);
}

}  // namespace

Unsqueeze::Unsqueeze(const torch::lazy::Value& input, int dim)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::unsqueeze), {input},
          [&]() { return NodeOutputShape(input, dim); },
          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
