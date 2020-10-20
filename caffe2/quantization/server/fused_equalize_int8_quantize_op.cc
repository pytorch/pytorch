#include "fused_equalize_int8_quantize_op.h"
#include "dnnlowp_op.h"

#include "caffe2/core/tensor_int8.h"
#include "caffe2/quantization/server/int8_gen_quant_params.h"
#include "caffe2_dnnlowp_utils.h"
#include "dnnlowp_partition.h"

namespace caffe2 {

template <typename T>
FusedEqualizeInt8QuantizeOp<T>::FusedEqualizeInt8QuantizeOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : Operator<CPUContext>(operator_def, ws),
      qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {}

template <typename T>
bool FusedEqualizeInt8QuantizeOp<T>::RunOnDevice() {
  // fuse broadcast multiplication of equalization vector with int8quant
  // the activations) and the weight
  using namespace dnnlowp;

  if (!arguments_parsed_) {
    dnnlowp::ParseDNNLowPOperatorArguments(this);
    arguments_parsed_ = true;
  }

  // get activation matrix and equalization vector
  CAFFE_ENFORCE(InputSize() <= 3);
  const auto& X = Input(0);
  const auto& S = Input(1);
  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(S.dim(), 1);

  const int64_t numRow = X.size_to_dim(1);
  const int64_t numCol = X.size_from_dim(1);

  CAFFE_ENFORCE_EQ(numCol, S.size_to_dim(1));
  CAFFE_ENFORCE(X.template IsType<float>());

  const float* X_data = X.template data<float>();
  const float* S_data = S.template data<float>();

  // quantized output
  int8::Int8TensorCPU* Xq =
      Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();
  Xq->t.ResizeLike(X);
  T* Xq_data = Xq->t.template mutable_data<T>();

  std::vector<float> Xeq(numCol);
  float* Xeq_data = Xeq.data();

  bool use_input_qparam = false;
  float in_scale = 0;
  int in_zero_point = 0;
  if (InputSize() == 3) {
    // if qparam is given
    use_input_qparam = true;
    const auto* input_qparam_blob =
        Input<caffe2::unique_ptr<Int8QuantParamsBlob>>(2).get();
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

  // fused equalization and quantization
  for (int64_t i = 0; i < numRow; ++i) {
    const int64_t idx = numCol * i;
    for (int64_t j = 0; j < numCol; ++j) {
      Xeq_data[j] = X_data[idx + j] * S_data[j];
    }
    fbgemm::Quantize<T>(Xeq_data, Xq_data + idx, numCol, in_qparams);
  }

  PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

  return true;
}

OPERATOR_SCHEMA(FusedEqualizeInt8Quantize)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Input(
        0,
        "X",
        "The input data, or last N samples of the output activations.")
    .Input(
        1,
        "S",
        "Equalized scale that will be multiplied to the columns of input.")
    .Output(0, "X_q", "Equalized and int8 quantized input data.")
    .SetDoc(R"DOC(
Given an activation matrix X and a vector of equalization parameter S,
return the equalized and quantized activation matrix X_q.

)DOC")
    .IdenticalTypeAndShapeOfInput(0);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FusedEqualizeInt8Quantize,
    DNNLOWP,
    FusedEqualizeInt8QuantizeOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FusedEqualizeInt8Quantize,
    DNNLOWP_ROWWISE,
    FusedEqualizeInt8QuantizeOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FusedEqualizeInt8Quantize,
    DNNLOWP_16,
    FusedEqualizeInt8QuantizeOp<uint16_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    FusedEqualizeInt8Quantize,
    DNNLOWP_ROWWISE_16,
    FusedEqualizeInt8QuantizeOp<uint16_t>);
} // namespace caffe2
