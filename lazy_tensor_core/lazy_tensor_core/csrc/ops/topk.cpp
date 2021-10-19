#include "lazy_tensor_core/csrc/ops/topk.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

TopK::TopK(const torch::lazy::Value& input, lazy_tensors::int64 k, lazy_tensors::int64 dim,
           bool largest, bool sorted)
    : TsNode(torch::lazy::OpKind(at::aten::topk), {input},
           /*num_outputs=*/2,
           torch::lazy::MHash(k, dim, largest, sorted)),
      k_(k),
      dim_(dim),
      largest_(largest),
      sorted_(sorted) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr TopK::Clone(OpList operands) const {
  return torch::lazy::MakeNode<TopK>(operands.at(0), k_, dim_, largest_, sorted_);
}

std::string TopK::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", k=" << k_ << ", dim=" << dim_
     << ", largest=" << largest_ << ", sorted=" << sorted_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
