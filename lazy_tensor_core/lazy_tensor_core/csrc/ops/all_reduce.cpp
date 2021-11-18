#include "lazy_tensor_core/csrc/ops/all_reduce.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

std::vector<torch::lazy::Value> GetOperandList(
    torch::lazy::OpList operands, const torch::lazy::Value& token) {
  std::vector<torch::lazy::Value> operand_list(operands.begin(), operands.end());
  operand_list.push_back(token);
  return operand_list;
}

}  // namespace

AllReduce::AllReduce(AllReduceType reduce_type, torch::lazy::OpList operands,
                     const torch::lazy::Value& token, double scale,
                     std::vector<std::vector<int64_t>> groups)
    : torch::lazy::TsNode(
          ltc_cross_replica_sum, GetOperandList(operands, token),
          /*num_outputs=*/operands.size() + 1,
          torch::lazy::MHash(lazy_tensors::util::GetEnumValue(reduce_type),
                             scale, groups)),
      reduce_type_(reduce_type),
      scale_(scale),
      groups_(std::move(groups)) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string AllReduce::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString()
     << ", reduce_type=" << lazy_tensors::util::GetEnumValue(reduce_type_)
     << ", scale=" << scale_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << c10::Join(", ", groups_[i]) << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
