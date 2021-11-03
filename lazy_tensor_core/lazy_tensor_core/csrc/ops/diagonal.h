#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Diagonal : public TsNode {
 public:
  Diagonal(const torch::lazy::Value& input, int64_t offset, int64_t dim1,
           int64_t dim2);

  std::string ToString() const override;

  int64_t offset() const { return offset_; }

  int64_t dim1() const { return dim1_; }

  int64_t dim2() const { return dim2_; }

  static lazy_tensors::Shape MakeDiagonalShape(const lazy_tensors::Shape& shape,
                                               int64_t offset, int64_t dim1,
                                               int64_t dim2);

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
