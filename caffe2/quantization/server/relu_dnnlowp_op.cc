#include "caffe2/operators/relu_op.h"

#include <limits>

#include "caffe2/core/tensor_int8.h"
#include "caffe2_dnnlowp_utils.h"

namespace caffe2 {

namespace {

template <typename T>
void ReluAVX2(const int N, const int zero_point, const T* X, T* Y);

template <>
void ReluAVX2<uint8_t>(
    const int N,
    const int zero_point,
    const uint8_t* X,
    uint8_t* Y) {
  constexpr int kVLen = 32;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  const __m256i zero_v = _mm256_set1_epi8(static_cast<uint8_t>(zero_point));
  for (int i = 0; i < n; i += kVLen) {
    __m256i cur_v =
        _mm256_max_epu8(_mm256_loadu_si256((const __m256i*)(X + i)), zero_v);
    _mm256_storeu_si256((__m256i*)(Y + i), cur_v);
  }
  for (int i = 0; i < r; ++i) {
    Y[n + i] = std::max(X[n + i], static_cast<uint8_t>(zero_point));
  }
}

template <>
void ReluAVX2<uint16_t>(
    const int N,
    const int zero_point,
    const uint16_t* X,
    uint16_t* Y) {
  constexpr int kVLen = 16;
  const int n = N / kVLen * kVLen;
  const int r = N % kVLen;
  const __m256i zero_v = _mm256_set1_epi16(static_cast<uint16_t>(zero_point));
  for (int i = 0; i < n; i += kVLen) {
    __m256i cur_v =
        _mm256_max_epu16(_mm256_loadu_si256((const __m256i*)(X + i)), zero_v);
    _mm256_storeu_si256((__m256i*)(Y + i), cur_v);
  }
  for (int i = 0; i < r; ++i) {
    Y[n + i] = std::max(X[n + i], static_cast<uint16_t>(zero_point));
  }
}

} // namespace

template <typename T>
class ReluDNNLowPOp final : public Operator<CPUContext> {
 public:
  ReluDNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) {}

  bool RunOnDevice() override;

 private:
  std::unique_ptr<dnnlowp::QuantizationFactory> qfactory_;
};

template <typename T>
bool ReluDNNLowPOp<T>::RunOnDevice() {
  auto& X = InputIsType<int8::Int8TensorCPU>(0)
      ? OperatorBase::Input<int8::Int8TensorCPU>(0).t
      : Input(0);

  TensorCPU* Y = nullptr;
  if (InputIsType<int8::Int8TensorCPU>(0)) {
    // The output follows the same type as input because ReLU can be inplace
    Y = &Outputs()[0]->template GetMutable<int8::Int8TensorCPU>()->t;
  } else {
    Y = Output(0);
  }
  Y->ResizeLike(X);

  using namespace dnnlowp;

  // Choose quantization params
  TensorQuantizationParams in_qparams =
      GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

  // Quantize input if needed
  std::vector<T> X_temp, Y_temp;
  const T* X_data =
      QuantizeInputIfNeeded(this, 0, in_qparams, X_temp, qfactory_.get());

  T* Y_data = nullptr;
  if (X.template IsType<T>()) {
    Y_data = Y->template mutable_data<T>();
  } else {
    Y_temp.resize(Y->numel());
    Y_data = Y_temp.data();
  }

  CAFFE_ENFORCE_GE(in_qparams.zero_point, std::numeric_limits<T>::lowest());
  CAFFE_ENFORCE_LE(in_qparams.zero_point, std::numeric_limits<T>::max());
  const int N = X.numel();
  if (in_qparams.zero_point == std::numeric_limits<T>::lowest()) {
    if (Y_data != X_data) {
      std::memcpy(Y_data, X_data, N * sizeof(T));
    }
  } else {
    if (GetCpuId().avx2()) {
      ReluAVX2<T>(N, in_qparams.zero_point, X_data, Y_data);
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int i = 0; i < N; ++i) {
        Y_data[i] = std::max(X_data[i], static_cast<T>(in_qparams.zero_point));
      }
    }
  }

  // Even if there is a pre-chosen quantization parameters for the output,
  // it is ignored because relu output quantization should be same as the
  // input.
  PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

  // If input was not quantized, output should be dequantized because ReLU
  // can be inplace.
  if (!X.template IsType<T>()) {
    Dequantize(
        Y_data, Y->template mutable_data<float>(), Y->numel(), in_qparams);
  }

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, DNNLOWP, ReluDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, DNNLOWP_16, ReluDNNLowPOp<uint16_t>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(Int8Relu, DNNLOWP, ReluDNNLowPOp<uint8_t>);

} // namespace caffe2
