#include "lazy_tensor_core/csrc/ops/index_put.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexPut::IndexPut(const torch::lazy::Value& base,
                   const torch::lazy::Value& indices, int64_t start_dim,
                   const torch::lazy::Value& values, bool accumulate)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::index_put), {base, indices, values},
          {torch::lazy::GetShapeFromTsValue(base)},
          /*num_outputs=*/1, torch::lazy::MHash(start_dim, accumulate)),
      start_dim_(start_dim),
      accumulate_(accumulate) {}

std::string IndexPut::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", start_dim=" << start_dim_
     << ", accumulate=" << accumulate_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
