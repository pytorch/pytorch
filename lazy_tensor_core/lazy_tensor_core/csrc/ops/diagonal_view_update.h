#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class DiagonalViewUpdate : public torch::lazy::TsNode {
 public:
  DiagonalViewUpdate(const torch::lazy::Value& target,
                     const torch::lazy::Value& input, int64_t offset,
                     int64_t dim1, int64_t dim2);

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
