#include "lazy_tensor_core/csrc/ops/index_along_dim.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexAlongDim::IndexAlongDim(torch::lazy::OpKind op,
                             const torch::lazy::Value& buffer,
                             const torch::lazy::Value& index,
                             const torch::lazy::Value& value, int64_t dim)
    : torch::lazy::TsNode(op, {buffer, index, value},
                          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string IndexAlongDim::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
