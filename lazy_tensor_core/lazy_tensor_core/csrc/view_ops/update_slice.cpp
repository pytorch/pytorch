#include "lazy_tensor_core/csrc/view_ops/update_slice.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

UpdateSlice::UpdateSlice(const torch::lazy::Value& input,
                         const torch::lazy::Value& source,
                         c10::ArrayRef<int64_t> base_indices)
    : torch::lazy::TsNode(ltc_update_slice, {input, source},
                          /*num_outputs=*/1, torch::lazy::MHash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {
  SetShapeDeferred([&]() { return compiler::InferShape(this); });
}

std::string UpdateSlice::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", base_indices=("
     << c10::Join(", ", base_indices_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
