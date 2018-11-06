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
    const OperatorDef& operator_def, Workspace *ws)
  : Operator<CPUContext>(operator_def, ws),
    qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {}

template <typename T>
bool QuantizeDNNLowPOp<T>::RunOnDevice() {
  using namespace dnnlowp;

  if (!arguments_parsed_) {
    dnnlowp::ParseDNNLowPOperatorArguments(this);
    arguments_parsed_ = true;
  }

  CAFFE_ENFORCE(Input(0).template IsType<float>());

  TensorQuantizationParams in_qparams;
  if (HasStaticQuantization(this)) {
    in_qparams = GetStaticQuantizationParamsOf(this, 0);
  } else {
    in_qparams = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());
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
    Quantize<T>(
        in_data + i_begin,
        out_data + i_begin,
        i_end - i_begin,
        in_qparams);
  }

  PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

  return true;
}

OPERATOR_SCHEMA(Quantize)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeAndShapeOfInput(0);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Quantize, DNNLOWP, QuantizeDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Quantize, DNNLOWP_ROWWISE, QuantizeDNNLowPOp<uint8_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Quantize, DNNLOWP_16, QuantizeDNNLowPOp<uint16_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Quantize, DNNLOWP_ROWWISE_16, QuantizeDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Int8Quantize, DNNLOWP, QuantizeDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
  Int8Quantize, DNNLOWP_ROWWISE, QuantizeDNNLowPOp<uint8_t>);

} // namespace caffe2
