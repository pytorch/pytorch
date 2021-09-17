#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Unselect : public Node {
 public:
  Unselect(const Value& target, const Value& source, lazy_tensors::int64 dim,
           lazy_tensors::int64 start, lazy_tensors::int64 end,
           lazy_tensors::int64 stride);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 dim() const { return dim_; }

  lazy_tensors::int64 start() const { return start_; }

  lazy_tensors::int64 end() const { return end_; }

  lazy_tensors::int64 stride() const { return stride_; }

 private:
  lazy_tensors::int64 dim_;
  lazy_tensors::int64 start_;
  lazy_tensors::int64 end_;
  lazy_tensors::int64 stride_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
