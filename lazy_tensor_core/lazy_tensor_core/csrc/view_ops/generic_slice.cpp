#include "lazy_tensor_core/csrc/view_ops/generic_slice.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

GenericSlice::GenericSlice(const torch::lazy::Value& input,
                           c10::ArrayRef<int64_t> base_indices,
                           c10::ArrayRef<int64_t> sizes)
    : BaseNode(generic_slice, {input},
               /*num_outputs=*/1, torch::lazy::MHash(base_indices, sizes)),
      base_indices_(base_indices.begin(), base_indices.end()),
      sizes_(sizes.begin(), sizes.end()) {
  SetShapeDeferred([&]() { return compiler::InferShape(this); });
}

std::string GenericSlice::ToString() const {
  std::stringstream ss;
  ss << BaseNode::ToString() << ", base_indices=("
     << c10::Join(", ", base_indices_) << "), sizes=("
     << c10::Join(", ", sizes_) << ")";
  return ss.str();
}

GenericSliceReverse::GenericSliceReverse(const torch::lazy::Value& input,
                                         const torch::lazy::Value& source,
                                         c10::ArrayRef<int64_t> base_indices)
    : BaseNode(generic_slice_reverse, {input, source},
               /*num_outputs=*/1, torch::lazy::MHash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {
  SetShapeDeferred([&]() { return compiler::InferShape(this); });
}

std::string GenericSliceReverse::ToString() const {
  std::stringstream ss;
  ss << BaseNode::ToString() << ", base_indices=("
     << c10::Join(", ", base_indices_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
