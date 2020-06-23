#include "quantize_dnnlowp_op.h"
#include "dnnlowp_op.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/tensor_int8.h"
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

  CAFFE_ENFORCE(InputSize() == 1 || InputSize() == 3);
  CAFFE_ENFORCE(Input(0).template IsType<float>());

  bool use_input_qparam = false;
  float in_scale = 0;
  int in_zero_point = 0;
  if (InputSize() == 3) {
    use_input_qparam = true;

    CAFFE_ENFORCE(Input(1).template IsType<float>());
    CAFFE_ENFORCE(Input(2).template IsType<int>());

    const auto& in_1 = Input(1);
    CAFFE_ENFORCE_EQ(in_1.numel(), 1);
    in_scale = *(in_1.template data<float>());

    const auto& in_2 = Input(2);
    CAFFE_ENFORCE_EQ(in_2.numel(), 1);
    in_zero_point = *(in_2.template data<int>());
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

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int i_begin, i_end;
    tie(i_begin, i_end) = Get1DPartition(
        Input(0).numel(), dnnlowp_get_num_threads(), dnnlowp_get_thread_num());
    fbgemm::Quantize<T>(
        in_data + i_begin, out_data + i_begin, i_end - i_begin, in_qparams);
  }

  PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

  return true;
}

OPERATOR_SCHEMA(Quantize)
    .NumInputs(1, 3)
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
