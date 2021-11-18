#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Nms : public torch::lazy::TsNode {
 public:
  Nms(const torch::lazy::Value& boxes, const torch::lazy::Value& scores,
      const torch::lazy::Value& score_threshold,
      const torch::lazy::Value& iou_threshold, int64_t output_size);

  std::string ToString() const override;

  int64_t output_size() const { return output_size_; }

 private:
  int64_t output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
