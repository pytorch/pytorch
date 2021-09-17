#include "lazy_tensor_core/csrc/ops/log_softmax_backward.h"

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LogSoftmaxBackward::LogSoftmaxBackward(const Value& grad_output,
                                       const Value& output,
                                       lazy_tensors::int64 dim)
    : Node(ir::OpKind(at::aten::_log_softmax_backward_data),
           {grad_output, output}, grad_output.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(dim)),
      dim_(dim) {}

NodePtr LogSoftmaxBackward::Clone(OpList operands) const {
  return MakeNode<LogSoftmaxBackward>(operands.at(0), operands.at(1), dim_);
}

std::string LogSoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
