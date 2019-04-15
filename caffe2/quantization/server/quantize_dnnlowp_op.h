#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"

namespace caffe2 {

template <typename T>
class QuantizeDNNLowPOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  QuantizeDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice() override;

 private:
  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;
  bool arguments_parsed_{false};
}; // class QuantizeDNNLowPOp

} // namespace caffe2
