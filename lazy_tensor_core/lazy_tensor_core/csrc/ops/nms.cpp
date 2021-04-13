#include "lazy_tensor_core/csrc/ops/nms.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Nms::Nms(const Value& boxes, const Value& scores, const Value& score_threshold,
         const Value& iou_threshold, lazy_tensors::int64 output_size)
    : Node(ltc_nms, {boxes, scores, score_threshold, iou_threshold},
           /*num_outputs=*/2, lazy_tensors::util::MHash(output_size)),
      output_size_(output_size) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Nms::Clone(OpList operands) const {
  return MakeNode<Nms>(operands.at(0), operands.at(1), operands.at(2),
                       operands.at(3), output_size_);
}

std::string Nms::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=" << output_size_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
