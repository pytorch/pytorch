#pragma once

#include "caffe2/operators/relu_op.h"

#include "caffe2/core/tensor_int8.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"

namespace caffe2 {

template <typename T>
class ReluDNNLowPOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  ReluDNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {}

  bool RunOnDevice() override;

 private:
  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;
};

namespace internal {

template <typename T>
void ReluAVX2(const int N, const int zero_point, const T* X, T* Y);

} // namespace internal

} // namespace caffe2
