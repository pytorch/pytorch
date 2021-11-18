#pragma once

#include "lazy_tensor_core/csrc/view_ops/opcode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Diagonal : public BaseNode {
 public:
  Diagonal(const torch::lazy::Value& input, int64_t offset, int64_t dim1,
           int64_t dim2);

  std::string ToString() const override;

  int64_t offset() const { return offset_; }

  int64_t dim1() const { return dim1_; }

  int64_t dim2() const { return dim2_; }

  static torch::lazy::Shape MakeDiagonalShape(const torch::lazy::Shape& shape,
                                              int64_t offset, int64_t dim1,
                                              int64_t dim2);

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

class DiagonalReverse : public BaseNode {
 public:
  DiagonalReverse(const torch::lazy::Value& target,
                  const torch::lazy::Value& input, int64_t offset, int64_t dim1,
                  int64_t dim2);

  std::string ToString() const override;

  int64_t offset() const { return offset_; }

  int64_t dim1() const { return dim1_; }

  int64_t dim2() const { return dim2_; }

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
