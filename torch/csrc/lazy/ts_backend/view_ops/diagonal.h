#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Diagonal : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::diagonal);
  }

  Diagonal(const Value& input, int64_t offset, int64_t dim1, int64_t dim2);

  bool CanBeReused(const Value& input, int64_t offset, int64_t dim1, int64_t dim2)
      const {
    size_t i = 0;
    return (
        operand(i++) == input && offset_ == offset && dim1_ == dim1 &&
        dim2_ == dim2);
  }

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
