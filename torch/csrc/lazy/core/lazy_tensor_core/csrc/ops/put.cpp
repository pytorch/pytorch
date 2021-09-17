#include "lazy_tensor_core/csrc/ops/put.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Put::Put(const Value& input, const Value& index, const Value& source,
         bool accumulate)
    : Node(ir::OpKind(at::aten::put), {input, index, source}, input.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(accumulate)),
      accumulate_(accumulate) {}

NodePtr Put::Clone(OpList operands) const {
  return MakeNode<Put>(operands.at(0), operands.at(1), operands.at(2),
                       accumulate_);
}

std::string Put::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", accumulate=" << accumulate_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
