#pragma once

#include "caffe2/operators/concat_split_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

template <typename T>
class ConcatDNNLowPOp final : public DNNLowPOp<T, ConcatOp<CPUContext>> {
 public:
  ConcatDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ConcatOp<CPUContext>);

 private:
  void GetQuantizationParameters_();

  int axis_;
  int add_axis_;
  // Input: a number of tensors. Output: Y, split
  // The split are stored in CPU.

  std::vector<dnnlowp::RequantizationParams> requantization_params_;
}; // class ConcatDNNLowPOp

} // namespace caffe2
