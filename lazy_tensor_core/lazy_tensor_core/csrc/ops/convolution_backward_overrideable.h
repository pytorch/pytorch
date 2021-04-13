#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/primitive_types.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ConvolutionBackwardOverrideable : public Node {
 public:
  ConvolutionBackwardOverrideable(
      const Value& grad_output, const Value& input, const Value& weight,
      std::vector<lazy_tensors::int64> stride,
      std::vector<lazy_tensors::int64> padding,
      std::vector<lazy_tensors::int64> dilation, bool transposed,
      std::vector<lazy_tensors::int64> output_padding,
      lazy_tensors::int64 groups);

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

 private:
  std::vector<lazy_tensors::int64> stride_;
  std::vector<lazy_tensors::int64> padding_;
  std::vector<lazy_tensors::int64> dilation_;
  std::vector<lazy_tensors::int64> output_padding_;
  bool transposed_;
  lazy_tensors::int64 groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
