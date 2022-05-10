#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Select : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::select);
  }

  Select(
      const Value& input,
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

  static int64_t GetStride(int64_t start, int64_t end, int64_t stride);

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

} // namespace lazy
} // namespace torch
