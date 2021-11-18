#pragma once

#include "lazy_tensor_core/csrc/view_ops/opcode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Select : public BaseNode {
 public:
  Select(const torch::lazy::Value& input, int64_t dim, int64_t start,
         int64_t end, int64_t stride);

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

  int64_t start() const { return start_; }

  int64_t end() const { return end_; }

  int64_t stride() const { return stride_; }

  static torch::lazy::Shape MakeSelectShape(const torch::lazy::Shape& shape,
                                            int64_t dim, int64_t start,
                                            int64_t end, int64_t stride);

  static int64_t GetStride(int64_t start, int64_t end, int64_t stride);

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

class SelectReverse : public BaseNode {
 public:
  SelectReverse(const torch::lazy::Value& target,
                const torch::lazy::Value& source, int64_t dim, int64_t start,
                int64_t end, int64_t stride);

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

  int64_t start() const { return start_; }

  int64_t end() const { return end_; }

  int64_t stride() const { return stride_; }

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
