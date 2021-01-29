#include "quantize_dnnlowp_op.h"
#include "dnnlowp_op.h"

#include "caffe2/core/tensor_int8.h"
#include "caffe2/quantization/server/int8_gen_quant_params.h"
#include "caffe2_dnnlowp_utils.h"
#include "dnnlowp_partition.h"

namespace caffe2 {

using namespace std;

template <typename T>
QuantizeDNNLowPOp<T>::QuantizeDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {}

template <typename T>
bool QuantizeDNNLowPOp<T>::RunOnDevice() {
  using namespace dnnlowp;

  if (!arguments_parsed_) {
    dnnlowp::ParseDNNLowPOperatorArguments(this);
    arguments_parsed_ = true;
  }

  CAFFE_ENFORCE(InputSize() <= 2);
  CAFFE_ENFORCE(Input(0).template IsType<float>());

  bool use_input_qparam = false;
  float in_scale = 0;
  int in_zero_point = 0;
  if (InputSize() == 2) {
    use_input_qparam = true;

    const auto* input_qparam_blob =
        Input<caffe2::unique_ptr<Int8QuantParamsBlob>>(1).get();
    CAFFE_ENFORCE(input_qparam_blob);
    in_scale = input_qparam_blob->qparam.scale;
    in_zero_point = input_qparam_blob->qparam.zero_point;
  }

  TensorQuantizationParams in_qparams;

  if (use_input_qparam) {
    in_qparams.scale = in_scale;
    in_qparams.zero_point = in_zero_point;
    in_qparams.precision = qfactory_->GetActivationPrecision();
  } else {
    if (HasStaticQuantization(this)) {
      in_qparams = GetStaticQuantizationParamsOf(this, 0);
    } else {
      in_qparams = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());
    }
  }

  int8::Int8TensorCPU* output =
      Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();
  output->t.ResizeLike(Input(0));

  const float* in_data = Input(0).template data<float>();
  T* out_data = output->t.template mutable_data<T>();

  fbgemm::Quantize<T>(in_data, out_data, Input(0).numel(), in_qparams);

  PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

  return true;
}

OPERATOR_SCHEMA(Quantize)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Quantize,
    DNNLOWP,
    QuantizeDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Quantize,
    DNNLOWP_ROWWISE,
    QuantizeDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Quantize,
    DNNLOWP_16,
    QuantizeDNNLowPOp<uint16_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Quantize,
    DNNLOWP_ROWWISE_16,
    QuantizeDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Quantize,
    DNNLOWP,
    QuantizeDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Quantize,
    DNNLOWP_ROWWISE,
    QuantizeDNNLowPOp<uint8_t>);

} // namespace caffe2
