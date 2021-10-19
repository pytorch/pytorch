#include "lazy_tensor_core/csrc/ops/diagonal_view_update.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DiagonalViewUpdate::DiagonalViewUpdate(const torch::lazy::Value& target, const torch::lazy::Value& input,
                                       lazy_tensors::int64 offset,
                                       lazy_tensors::int64 dim1,
                                       lazy_tensors::int64 dim2)
    : TsNode(ltc_diagonal_view_update, {target, input}, ir::GetShapeFromTsValue(target),
           /*num_outputs=*/1, torch::lazy::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

NodePtr DiagonalViewUpdate::Clone(OpList operands) const {
  return torch::lazy::MakeNode<DiagonalViewUpdate>(operands.at(0), operands.at(1), offset_,
                                      dim1_, dim2_);
}

std::string DiagonalViewUpdate::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
