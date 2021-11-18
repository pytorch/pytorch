#include "lazy_tensor_core/csrc/ops/collective_permute.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

CollectivePermute::CollectivePermute(
    const torch::lazy::Value& input, const torch::lazy::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs)
    : torch::lazy::TsNode(ltc_collective_permute, {input, token},
                          /*num_outputs=*/2,
                          torch::lazy::MHash(source_target_pairs)),
      source_target_pairs_(std::move(source_target_pairs)) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string CollectivePermute::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", source_target_pairs=(";
  for (size_t i = 0; i < source_target_pairs_.size(); ++i) {
    ss << (i == 0 ? "(" : ", (");
    ss << source_target_pairs_[i].first << ", "
       << source_target_pairs_[i].second << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
