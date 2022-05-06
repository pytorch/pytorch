#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/ts_backend/view_ops/squeeze.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch {
namespace lazy {

const OpKind Squeeze::class_op_kind(at::aten::squeeze);

Squeeze::Squeeze(const torch::lazy::Value& input, int dim)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::squeeze), {input},
                          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {
  addComputedShape(
      [&]() {
        const auto& input_shape = input.shape();
        return torch::lazy::Shape(input_shape.scalar_type(),
          BuildSqueezedDimensions(input_shape.sizes(), dim));
      });
}

std::string Squeeze::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace lazy
}  // namespace torch
