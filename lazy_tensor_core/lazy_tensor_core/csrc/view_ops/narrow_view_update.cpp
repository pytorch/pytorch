#include "lazy_tensor_core/csrc/view_ops/narrow_view_update.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NarrowViewUpdate::NarrowViewUpdate(const torch::lazy::Value& input,
                                   const torch::lazy::Value& source,
                                   c10::ArrayRef<int64_t> base_indices)
    : torch::lazy::TsNode(ltc_narrow_view_update, {input, source},
                          /*num_outputs=*/1, torch::lazy::MHash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {
  SetShapeDeferred(
      [&]() { return torch::lazy::GetShapeFromTsOutput(operand(0)); });
}

std::string NarrowViewUpdate::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", base_indices=("
     << c10::Join(", ", base_indices_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
