#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Nms : public Node {
 public:
  Nms(const Value& boxes, const Value& scores, const Value& score_threshold,
      const Value& iou_threshold, lazy_tensors::int64 output_size);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 output_size() const { return output_size_; }

 private:
  lazy_tensors::int64 output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
