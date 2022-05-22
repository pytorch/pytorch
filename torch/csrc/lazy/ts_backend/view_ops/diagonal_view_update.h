#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>
#include <lazy/core/internal_ops/ltc_ops.h>

namespace torch {
namespace lazy {

class TORCH_API DiagonalViewUpdate : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return ltc_diagonal_view_update;
  }

  DiagonalViewUpdate(
      const Value& target,
      const Value& input,
      int64_t offset,
      int64_t dim1,
      int64_t dim2);

  std::string ToString() const override;

  int64_t offset() const {
    return offset_;
  }

  int64_t dim1() const {
    return dim1_;
  }

  int64_t dim2() const {
    return dim2_;
  }

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

} // namespace lazy
} // namespace torch
