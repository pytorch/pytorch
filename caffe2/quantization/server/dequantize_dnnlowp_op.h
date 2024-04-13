#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/quantization/server/dnnlowp.h"

namespace caffe2 {

template <typename T>
class DequantizeDNNLowPOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  DequantizeDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice() override;

 private:
  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;
}; // class DequantizeDNNLowPOp

} // namespace caffe2
