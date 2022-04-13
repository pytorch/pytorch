#include <torch/csrc/lazy/core/view_ops/narrow_view_update.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>

namespace torch {
namespace lazy {

NarrowViewUpdate::NarrowViewUpdate(
    const Value& input,
    const Value& source,
    c10::ArrayRef<int64_t> base_indices)
    : TsNode(
          ltc_narrow_view_update,
          {input, source},
          /*num_outputs=*/1,
          MHash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {
  SetShapeDeferred([&]() { return operand(0).shape(); });
}

std::string NarrowViewUpdate::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", base_indices=("
     << c10::Join(", ", base_indices_) << ")";
  return ss.str();
}

} // namespace lazy
} // namespace torch
