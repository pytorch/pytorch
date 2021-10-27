#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AvgPoolNd : public TsNode {
 public:
  AvgPoolNd(const torch::lazy::Value& input, int64_t spatial_dim_count,
            std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
            std::vector<int64_t> padding, bool ceil_mode,
            bool count_include_pad);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  int64_t spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<int64_t>& kernel_size() const { return kernel_size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  const std::vector<int64_t>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

  bool count_include_pad() const { return count_include_pad_; }

 private:
  int64_t spatial_dim_count_;
  // The parameters of the pooling.
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  bool ceil_mode_;
  // Whether the counts used to compute the average should include the added
  // padding.
  bool count_include_pad_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
