#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ConvolutionBackwardOverrideable : public TsNode {
 public:
  ConvolutionBackwardOverrideable(
      const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
      const torch::lazy::Value& weight, std::vector<int64_t> stride,
      std::vector<int64_t> padding, std::vector<int64_t> dilation,
      bool transposed, std::vector<int64_t> output_padding, int64_t groups);

  ConvolutionBackwardOverrideable(
      const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
      const torch::lazy::Value& weight, std::vector<int64_t> stride,
      std::vector<int64_t> padding, std::vector<int64_t> dilation,
      bool transposed, std::vector<int64_t> output_padding, int64_t groups,
      std::array<bool, 3> output_mask);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& stride() const { return stride_; }

  const std::vector<int64_t>& padding() const { return padding_; }

  const std::vector<int64_t>& dilation() const { return dilation_; }

  bool transposed() const { return transposed_; }

  const std::vector<int64_t>& output_padding() const { return output_padding_; }

  int64_t groups() const { return groups_; }

  std::array<bool, 3> output_mask() const { return output_mask_; }

 private:
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> output_padding_;
  bool transposed_;
  int64_t groups_;
  std::array<bool, 3> output_mask_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
