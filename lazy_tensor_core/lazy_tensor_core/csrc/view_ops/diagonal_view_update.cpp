#include "lazy_tensor_core/csrc/view_ops/diagonal_view_update.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DiagonalViewUpdate::DiagonalViewUpdate(const torch::lazy::Value& target,
                                       const torch::lazy::Value& input,
                                       int64_t offset, int64_t dim1,
                                       int64_t dim2)
    : torch::lazy::TsNode(ltc_diagonal_view_update, {target, input},
                          {torch::lazy::GetShapeFromTsValue(target)},
                          /*num_outputs=*/1,
                          torch::lazy::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

std::string DiagonalViewUpdate::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", offset=" << offset_
     << ", dim1=" << dim1_ << ", dim2=" << dim2_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
