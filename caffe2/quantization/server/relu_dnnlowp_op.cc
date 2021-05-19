#include "caffe2/quantization/server/relu_dnnlowp_op.h"

#include <limits>

namespace caffe2 {

template <typename T>
bool ReluDNNLowPOp<T>::RunOnDevice() {
  auto& X = InputIsType<int8::Int8TensorCPU>(0)
      ? (this->template Input<int8::Int8TensorCPU>(0)).t
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
  const T* X_data = QuantizeInputIfNeeded(this, 0, in_qparams, X_temp);

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
      internal::ReluAVX2<T>(N, in_qparams.zero_point, X_data, Y_data);
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
    fbgemm::Dequantize<T>(
        Y_data, Y->template mutable_data<float>(), Y->numel(), in_qparams);
  }

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, DNNLOWP, ReluDNNLowPOp<uint8_t>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(Relu, DNNLOWP_16, ReluDNNLowPOp<uint16_t>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR_WITH_ENGINE(Int8Relu, DNNLOWP, ReluDNNLowPOp<uint8_t>);

} // namespace caffe2
