#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Diagonal : public TsNode {
 public:
  Diagonal(const Value& input, int64_t offset, int64_t dim1, int64_t dim2);

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

  static Shape MakeDiagonalShape(
      const Shape& shape,
      int64_t offset,
      int64_t dim1,
      int64_t dim2);

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

} // namespace lazy
} // namespace torch
