#include "utility_dnnlowp_ops.h"

namespace caffe2 {

template <typename T>
GatherDNNLowPOp<T>::GatherDNNLowPOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : GatherOp<CPUContext>(operator_def, ws),
      qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {}

template <typename T>
GatherDNNLowPOp<T>::~GatherDNNLowPOp() {
  if (measure_quantization_error_) {
    dnnlowp::ReportQuantizationError(this, quantization_error_stats_);
  }
}

template <typename T>
bool GatherDNNLowPOp<T>::RunOnDevice() {
  using namespace dnnlowp;

  if (!arguments_parsed_) {
    dnnlowp::ParseDNNLowPOperatorArguments(
        this, &dequantize_output_, &measure_quantization_error_);
    arguments_parsed_ = true;
  }

  if (!InputIsType<int8::Int8TensorCPU>(DATA)) {
    if (dequantize_output_) {
      return GatherOp<CPUContext>::RunOnDevice();
    } else {
      // If input or output is float, delegate to fp32 op
      Fp32Op_()->DequantizeInput();
      // dequantize input if it's not already float
      if (!Fp32Op_()->Get()->RunOnDevice()) {
        return false;
      }

      int8::Int8TensorCPU* output =
          Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();

      output->t.ResizeLike(*Fp32Op_()->Get()->Output(0));
      T* out_data = output->t.template mutable_data<T>();

      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      TensorQuantizationParams out_qparams;
      if (HasStaticQuantization(this)) {
        out_qparams = GetStaticQuantizationParamsOf(this, 0);
      } else {
        out_qparams = Fp32Op_()->GetOutputQuantizationParams(qfactory_.get());
      }

      fbgemm::Quantize<T>(
          static_cast<const float*>(Fp32Op_()->Get()->Output(0)->raw_data()),
          out_data,
          output->t.numel(),
          out_qparams);

      PropagateOutputTensorQuantizationParams(this, 0, out_qparams);
    }
  } else {
    DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(INDICES));

    TensorQuantizationParams in_qparams =
        GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

    PropagateOutputTensorQuantizationParams(this, 0, in_qparams);
  }

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(Gather, DNNLOWP, GatherDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Gather,
    DNNLOWP,
    GatherDNNLowPOp<uint8_t>);

} // namespace caffe2
