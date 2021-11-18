#include "lazy_tensor_core/csrc/ops/all_to_all.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AllToAll::AllToAll(const torch::lazy::Value& input,
                   const torch::lazy::Value& token, int64_t split_dimension,
                   int64_t concat_dimension, int64_t split_count,
                   std::vector<std::vector<int64_t>> groups)
    : torch::lazy::TsNode(ltc_all_to_all, {input, token},
                          /*num_outputs=*/2,
                          torch::lazy::MHash(split_dimension, concat_dimension,
                                             split_count, groups)),
      split_dimension_(split_dimension),
      concat_dimension_(concat_dimension),
      split_count_(split_count),
      groups_(std::move(groups)) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string AllToAll::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString()
     << ", split_dimension=" << split_dimension_
     << ", concat_dimension=" << concat_dimension_
     << ", split_count=" << split_count_ << ", groups=(";
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
