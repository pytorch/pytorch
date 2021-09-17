#include "lazy_tensor_core/csrc/ops/nll_loss_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NllLossBackward::NllLossBackward(const Value& grad_output, const Value& logits,
                                 const Value& labels,
                                 const c10::optional<Value>& weight,
                                 const c10::optional<Value>& total_weight,
                                 ReductionMode reduction, int ignore_index)
    : Node(ir::OpKind(at::aten::nll_loss_backward),
           lazy_tensors::util::GetValuesVector<Value>(
               {grad_output, logits, labels}, {&weight, &total_weight}),
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(
               lazy_tensors::util::GetEnumValue(reduction), ignore_index)),
      reduction_(reduction),
      ignore_index_(ignore_index) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr NllLossBackward::Clone(OpList operands) const {
  c10::optional<Value> weight;
  c10::optional<Value> total_weight;
  if (operands.size() > 3) {
    weight = operands.at(3);
    total_weight = operands.at(4);
  }
  return MakeNode<NllLossBackward>(operands.at(0), operands.at(1),
                                   operands.at(2), weight, total_weight,
                                   reduction_, ignore_index_);
}

std::string NllLossBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << lazy_tensors::util::GetEnumValue(reduction_)
     << ", ignore_index=" << ignore_index_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
