#include "lazy_tensor_core/csrc/ops/nll_loss_forward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NllLossForward::NllLossForward(const torch::lazy::Value& logits, const torch::lazy::Value& labels,
                               const c10::optional<torch::lazy::Value>& weight,
                               ReductionMode reduction, int ignore_index)
    : TsNode(torch::lazy::OpKind(at::aten::nll_loss_forward),
             lazy_tensors::util::GetValuesVector<torch::lazy::Value>({logits, labels},
                                                        {&weight}),
             /*num_outputs=*/2,
             torch::lazy::MHash(lazy_tensors::util::GetEnumValue(reduction),
                                ignore_index)),
      reduction_(reduction),
      ignore_index_(ignore_index) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr NllLossForward::Clone(OpList operands) const {
  c10::optional<torch::lazy::Value> weight;
  if (operands.size() > 2) {
    weight = operands.at(2);
  }
  return torch::lazy::MakeNode<NllLossForward>(operands.at(0), operands.at(1), weight,
                                  reduction_, ignore_index_);
}

std::string NllLossForward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString()
     << ", reduction=" << lazy_tensors::util::GetEnumValue(reduction_)
     << ", ignore_index=" << ignore_index_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
