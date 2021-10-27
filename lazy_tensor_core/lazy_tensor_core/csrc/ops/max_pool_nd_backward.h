#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class MaxPoolNdBackward : public TsNode {
 public:
  MaxPoolNdBackward(const torch::lazy::Value& grad_output,
                    const torch::lazy::Value& input, int64_t spatial_dim_count,
                    std::vector<int64_t> kernel_size,
                    std::vector<int64_t> stride, std::vector<int64_t> padding,
                    bool ceil_mode);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  int64_t spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<int64_t>& kernel_size() const { return kernel_size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  const std::vector<int64_t>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

 private:
  int64_t spatial_dim_count_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  bool ceil_mode_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
