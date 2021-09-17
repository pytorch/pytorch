#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MaxPoolNd : public Node {
 public:
  MaxPoolNd(const Value& input, lazy_tensors::int64 spatial_dim_count,
            std::vector<lazy_tensors::int64> kernel_size,
            std::vector<lazy_tensors::int64> stride,
            std::vector<lazy_tensors::int64> padding, bool ceil_mode);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<lazy_tensors::int64>& kernel_size() const {
    return kernel_size_;
  }

  const std::vector<lazy_tensors::int64>& stride() const { return stride_; }

  const std::vector<lazy_tensors::int64>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

 private:
  lazy_tensors::int64 spatial_dim_count_;
  // The parameters of the pooling. Only support the same kernel size, stride
  // and padding in both dimensions for now.
  std::vector<lazy_tensors::int64> kernel_size_;
  std::vector<lazy_tensors::int64> stride_;
  std::vector<lazy_tensors::int64> padding_;
  bool ceil_mode_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
