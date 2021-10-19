#include "lazy_tensor_core/csrc/ops/log_base.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LogBase::LogBase(const torch::lazy::Value& input, torch::lazy::OpKind kind, double base)
    : TsNode(kind, {input}, ir::GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(base)),
      base_(base) {}

NodePtr LogBase::Clone(OpList operands) const {
  return torch::lazy::MakeNode<LogBase>(operands.at(0), op(), base_);
}

std::string LogBase::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", base=" << base_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
