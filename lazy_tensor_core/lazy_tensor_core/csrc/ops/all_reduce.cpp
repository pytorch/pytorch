#include "lazy_tensor_core/csrc/ops/all_reduce.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

std::vector<torch::lazy::Value> GetOperandList(OpList operands,
                                  const torch::lazy::Value& token) {
  std::vector<torch::lazy::Value> operand_list(operands.begin(), operands.end());
  operand_list.push_back(token);
  return operand_list;
}

}  // namespace

AllReduce::AllReduce(AllReduceType reduce_type,
                     OpList operands,
                     const torch::lazy::Value& token, double scale,
                     std::vector<std::vector<lazy_tensors::int64>> groups)
    : TsNode(ltc_cross_replica_sum, GetOperandList(operands, token),
           /*num_outputs=*/operands.size() + 1,
           torch::lazy::MHash(
               lazy_tensors::util::GetEnumValue(reduce_type), scale, groups)),
      reduce_type_(reduce_type),
      scale_(scale),
      groups_(std::move(groups)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr AllReduce::Clone(OpList operands) const {
  std::vector<torch::lazy::Value> operand_list(operands.begin(), operands.end() - 1);
  return torch::lazy::MakeNode<AllReduce>(reduce_type_, operand_list, operands.back(),
                             scale_, groups_);
}

std::string AllReduce::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString()
     << ", reduce_type=" << lazy_tensors::util::GetEnumValue(reduce_type_)
     << ", scale=" << scale_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << lazy_tensors::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
