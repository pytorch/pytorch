#include "lazy_tensor_core/csrc/ops/replication_pad.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ReplicationPad::ReplicationPad(const Value& input,
                               std::vector<lazy_tensors::int64> padding)
    : Node(ltc_replication_pad, {input},
           /*num_outputs=*/1, lazy_tensors::util::MHash(padding)),
      padding_(std::move(padding)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr ReplicationPad::Clone(OpList operands) const {
  return MakeNode<ReplicationPad>(operands.at(0), padding_);
}

std::string ReplicationPad::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", padding=("
     << lazy_tensors::StrJoin(padding_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
