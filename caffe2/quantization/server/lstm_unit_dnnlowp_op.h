#pragma once

#include "caffe2/operators/lstm_unit_op.h"
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"
#include "caffe2/quantization/server/op_wrapper.h"
#include "caffe2/quantization/server/sigmoid.h"

namespace caffe2 {

template <typename T>
class LSTMUnitDNNLowPOp final : public LSTMUnitOp<CPUContext> {
  static_assert(std::is_integral<T>::value, "Integral required.");

 public:
  LSTMUnitDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);
  ~LSTMUnitDNNLowPOp() override;
  bool RunOnDevice() override;

 private:
  const TensorCPU& InputTensorCPU_(int idx);
  TensorCPU* OutputTensorCPU_(int idx);
  bool GetQuantizationParameters_();
  OpWrapper<LSTMUnitOp<CPUContext>, T>* Fp32Op_();

  bool drop_states_;
  dnnlowp::Sigmoid<T> sigmoid_;
  dnnlowp::Tanh<T> tanh_;

  dnnlowp::TensorQuantizationParams H_in_qparams_, C_in_qparams_, G_in_qparams_,
      H_out_qparams_, C_out_qparams_;

  std::unique_ptr<OpWrapper<LSTMUnitOp<CPUContext>, T>> fp32_op_;
  bool dequantize_output_{false}, measure_quantization_error_{false};

  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;

  dnnlowp::QuantizationErrorStats cell_quantization_error_stats_,
      hidden_quantization_error_stats_;

  bool arguments_parsed_{false};
}; // class LSTMUnitDNNLowPOp

} // namespace caffe2
