#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"

namespace caffe2 {

template <typename T>
class FusedEqualizeInt8QuantizeOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  FusedEqualizeInt8QuantizeOp(const OperatorDef& operator_def, Workspace* ws);

  bool RunOnDevice() override;

 private:
  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;
  bool arguments_parsed_{false};
}; // class FusedMulInt8QuantizeOp

} // namespace caffe2
