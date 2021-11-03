#include "lazy_tensor_core/csrc/ops/index_get.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexGet::IndexGet(const torch::lazy::Value& base,
                   const torch::lazy::Value& indices, int64_t start_dim)
    : TsNode(OpKind(at::aten::index), {base, indices},
             /*num_outputs=*/1, torch::lazy::MHash(start_dim)),
      start_dim_(start_dim) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string IndexGet::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", start_dim=" << start_dim_;
  return ss.str();
}

NodePtr IndexGet::Clone(OpList operands) const {
  return torch::lazy::MakeNode<IndexGet>(operands.at(0), operands.at(1), start_dim_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
