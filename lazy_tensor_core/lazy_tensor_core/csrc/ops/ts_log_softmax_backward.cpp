#include "lazy_tensor_core/csrc/ops/ts_log_softmax_backward.h"

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

TSLogSoftmaxBackward::TSLogSoftmaxBackward(const Value& grad_output,
    const Value& output, lazy_tensors::int64 dim, const Value& self)
    : Node(ir::OpKind(at::aten::_log_softmax_backward_data),
          {grad_output, output, self}, grad_output.shape(),
          /*num_outputs=*/1, lazy_tensors::util::MHash(dim)),
      dim_(dim) {}

NodePtr TSLogSoftmaxBackward::Clone(OpList operands) const {
  return MakeNode<TSLogSoftmaxBackward>(operands.at(0), operands.at(1), dim_,
    operands.at(2));
}

std::string TSLogSoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
