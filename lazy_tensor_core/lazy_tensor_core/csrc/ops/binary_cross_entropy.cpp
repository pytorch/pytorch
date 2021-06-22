#include "lazy_tensor_core/csrc/ops/binary_cross_entropy.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

BinaryCrossEntropy::BinaryCrossEntropy(const Value& logits, const Value& labels,
                                       const c10::optional<Value>& weight,
                                       ReductionMode reduction)
    : Node(ir::OpKind(at::aten::binary_cross_entropy),
           lazy_tensors::util::GetValuesVector<Value>({logits, labels},
                                                      {&weight}),
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(
               lazy_tensors::util::GetEnumValue(reduction))),
      reduction_(reduction) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr BinaryCrossEntropy::Clone(OpList operands) const {
  c10::optional<Value> weight;
  if (operands.size() > 2) {
    weight = operands.at(2);
  }
  return MakeNode<BinaryCrossEntropy>(operands.at(0), operands.at(1), weight,
                                      reduction_);
}

std::string BinaryCrossEntropy::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << lazy_tensors::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
