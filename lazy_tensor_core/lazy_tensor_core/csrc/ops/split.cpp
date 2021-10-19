#include "lazy_tensor_core/csrc/ops/split.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Split::Split(const torch::lazy::Value& input, std::vector<lazy_tensors::int64> split_sizes,
             lazy_tensors::int64 dim)
    : TsNode(torch::lazy::OpKind(at::aten::split), {input},
           ComputeSplitCount(ir::GetShapeFromTsValue(input).dimensions(dim), split_sizes),
           torch::lazy::MHash(split_sizes, dim)),
      split_sizes_(std::move(split_sizes)),
      dim_(dim) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Split::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Split>(operands.at(0), split_sizes_, dim_);
}

std::string Split::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", split_sizes=("
     << lazy_tensors::StrJoin(split_sizes_, ", ") << "), dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
