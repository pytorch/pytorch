#include "dequantize_dnnlowp_op.h"

#include "caffe2/core/tensor_int8.h"
#include "caffe2_dnnlowp_utils.h"

namespace caffe2 {

template <typename T>
DequantizeDNNLowPOp<T>::DequantizeDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {
  if (this->debug_def().engine() == "DNNLOWP_16" ||
      this->debug_def().engine() == "DNNLOWP_ROWWISE_16") {
    LOG(WARNING)
        << this->debug_def().engine()
        << " is an experimental feature mostly for testing accuracy with "
           "fixed-point precision higher than 8 and performance is very slow";
  }
}

template <typename T>
bool DequantizeDNNLowPOp<T>::RunOnDevice() {
  using namespace dnnlowp;
  TensorQuantizationParams in_qparams =
      GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  const TensorCPU& input = InputIsType<int8::Int8TensorCPU>(0)
      ? this->template Input<int8::Int8TensorCPU>(0).t
      : Input(0);

  CAFFE_ENFORCE(input.template IsType<T>());
  Output(0)->ResizeLike(input);
  fbgemm::Dequantize<T>(
      input.template data<T>(),
      Output(0)->template mutable_data<float>(),
      input.numel(),
      in_qparams);

  return true;
}

OPERATOR_SCHEMA(Dequantize)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Dequantize,
    DNNLOWP,
    DequantizeDNNLowPOp<std::uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Dequantize,
    DNNLOWP_ROWWISE,
    DequantizeDNNLowPOp<std::uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Dequantize,
    DNNLOWP_16,
    DequantizeDNNLowPOp<std::uint16_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Dequantize,
    DNNLOWP_ROWWISE_16,
    DequantizeDNNLowPOp<std::uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Dequantize,
    DNNLOWP,
    DequantizeDNNLowPOp<std::uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Dequantize,
    DNNLOWP_ROWWISE,
    DequantizeDNNLowPOp<std::uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8DequantizeRowWise,
    DNNLOWP,
    DequantizeDNNLowPOp<std::uint8_t>);

} // namespace caffe2
