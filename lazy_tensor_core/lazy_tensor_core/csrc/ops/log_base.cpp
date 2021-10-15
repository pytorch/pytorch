#include "lazy_tensor_core/csrc/ops/log_base.h"


namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LogBase::LogBase(const Value& input, ir::OpKind kind, double base)
    : TsNode(kind, {input}, GetShapeFromTsValue(input),
           /*num_outputs=*/1, torch::lazy::MHash(base)),
      base_(base) {}

NodePtr LogBase::Clone(OpList operands) const {
  return MakeNode<LogBase>(operands.at(0), op(), base_);
}

std::string LogBase::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", base=" << base_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
