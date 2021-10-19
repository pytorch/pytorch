#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensors/primitive_types.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ConvolutionBackwardOverrideable : public TsNode {
 public:
  ConvolutionBackwardOverrideable(
      const torch::lazy::Value& grad_output, const torch::lazy::Value& input, const torch::lazy::Value& weight,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding,
      std::vector<lazy_tensors::int64> dilation, bool transposed,
      std::vector<lazy_tensors::int64> output_padding,
      lazy_tensors::int64 groups);

  ConvolutionBackwardOverrideable(
      const torch::lazy::Value& grad_output, const torch::lazy::Value& input, const torch::lazy::Value& weight,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding,
      std::vector<lazy_tensors::int64> dilation, bool transposed,
      std::vector<lazy_tensors::int64> output_padding,
      lazy_tensors::int64 groups, std::array<bool, 3> output_mask);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& stride() const { return stride_; }

  const std::vector<lazy_tensors::int64>& padding() const { return padding_; }

  const std::vector<lazy_tensors::int64>& dilation() const { return dilation_; }

  bool transposed() const { return transposed_; }

  const std::vector<lazy_tensors::int64>& output_padding() const {
    return output_padding_;
  }

  lazy_tensors::int64 groups() const { return groups_; }

  std::array<bool, 3> output_mask() const { return output_mask_; }

 private:
  std::vector<lazy_tensors::int64> stride_;
  std::vector<lazy_tensors::int64> padding_;
  std::vector<lazy_tensors::int64> dilation_;
  std::vector<lazy_tensors::int64> output_padding_;
  bool transposed_;
  lazy_tensors::int64 groups_;
  std::array<bool, 3> output_mask_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
