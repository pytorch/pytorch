#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <lazy/core/internal_ops/ltc_ops.h>

namespace torch {
namespace lazy {

class TORCH_API SelectViewUpdate : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return ltc_select_view_update;
  }

  SelectViewUpdate(
      const Value& target,
      const Value& source,
      int64_t dim,
      int64_t start,
      int64_t end,
      int64_t stride);

  std::string ToString() const override;

  int64_t dim() const {
    return dim_;
  }

  int64_t start() const {
    return start_;
  }

  int64_t end() const {
    return end_;
  }

  int64_t stride() const {
    return stride_;
  }

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

} // namespace lazy
} // namespace torch
