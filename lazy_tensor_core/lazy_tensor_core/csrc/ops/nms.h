#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Nms : public TsNode {
 public:
  Nms(const torch::lazy::Value& boxes, const torch::lazy::Value& scores, const torch::lazy::Value& score_threshold,
      const torch::lazy::Value& iou_threshold, lazy_tensors::int64 output_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 output_size() const { return output_size_; }

 private:
  lazy_tensors::int64 output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
