#include "lazy_tensor_core/csrc/ops/log_base.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LogBase::LogBase(const Value& input, ir::OpKind kind, double base)
    : Node(kind, {input}, input.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(base)),
      base_(base) {}

NodePtr LogBase::Clone(OpList operands) const {
  return MakeNode<LogBase>(operands.at(0), op(), base_);
}

std::string LogBase::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", base=" << base_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
