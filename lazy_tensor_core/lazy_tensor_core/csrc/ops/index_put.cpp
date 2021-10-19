#include "lazy_tensor_core/csrc/ops/index_put.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexPut::IndexPut(const torch::lazy::Value& base, const torch::lazy::Value& indices,
                   lazy_tensors::int64 start_dim, const torch::lazy::Value& values,
                   bool accumulate)
    : TsNode(OpKind(at::aten::index_put), {base, indices, values}, ir::GetShapeFromTsValue(base),
           /*num_outputs=*/1, torch::lazy::MHash(start_dim, accumulate)),
      start_dim_(start_dim),
      accumulate_(accumulate) {}

std::string IndexPut::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", start_dim=" << start_dim_
     << ", accumulate=" << accumulate_;
  return ss.str();
}

NodePtr IndexPut::Clone(OpList operands) const {
  return torch::lazy::MakeNode<IndexPut>(operands.at(0), operands.at(1), start_dim_,
                            operands.at(2), accumulate_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
