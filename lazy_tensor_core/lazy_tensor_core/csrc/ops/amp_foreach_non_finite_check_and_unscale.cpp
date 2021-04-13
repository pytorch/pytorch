#include "lazy_tensor_core/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

std::vector<Value> GetOperandList(lazy_tensors::Span<const Value> operands,
                                  const Value& found_inf,
                                  const Value& inv_scale) {
  std::vector<Value> operand_list(operands.begin(), operands.end());
  operand_list.push_back(found_inf);
  operand_list.push_back(inv_scale);
  return operand_list;
}

}  // namespace

AmpForachNonFiniteCheckAndUnscale::AmpForachNonFiniteCheckAndUnscale(
    const OpList& inputs, const Value& found_inf, const Value& inv_scale)
    : Node(ir::OpKind(at::aten::_amp_foreach_non_finite_check_and_unscale_),
           GetOperandList(inputs, found_inf, inv_scale),
           /*num_outputs=*/inputs.size() + 1) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr AmpForachNonFiniteCheckAndUnscale::Clone(OpList operands) const {
  std::vector<Value> operand_list(operands.begin(), operands.end() - 2);
  size_t sz = operand_list.size();
  return MakeNode<AmpForachNonFiniteCheckAndUnscale>(operand_list, operands[sz],
                                                     operands[sz + 1]);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
