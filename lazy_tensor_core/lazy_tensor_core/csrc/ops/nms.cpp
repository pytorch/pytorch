#include "lazy_tensor_core/csrc/ops/nms.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Nms::Nms(const torch::lazy::Value& boxes, const torch::lazy::Value& scores,
         const torch::lazy::Value& score_threshold,
         const torch::lazy::Value& iou_threshold, int64_t output_size)
    : torch::lazy::TsNode(ltc_nms,
                          {boxes, scores, score_threshold, iou_threshold},
                          /*num_outputs=*/2, torch::lazy::MHash(output_size)),
      output_size_(output_size) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string Nms::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", output_size=" << output_size_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
