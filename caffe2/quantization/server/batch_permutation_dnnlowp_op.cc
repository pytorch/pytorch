#include "caffe2/quantization/server/batch_permutation_dnnlowp_op.h"

namespace caffe2 {

template <typename T>
bool BatchPermutationDNNLowPOp<T>::RunOnDevice() {
  using namespace dnnlowp;

  this->ParseDNNLowPOperatorArguments_();

  // Choose quantization params
  in_qparams_[INPUT] =
      GetInputTensorQuantizationParamsOf(this, INPUT, qfactory_.get());

  const auto& X = InputTensorCPU_(INPUT);
  const auto& indices = Input(INDICES);
  auto* Y = OutputTensorCPU_(OUTPUT);

  if (X.dim32(0) == 0) {
    return true;
  }

  CAFFE_ENFORCE(indices.ndim() == 1, "indices must be 1-d");
  CAFFE_ENFORCE(
      X.dim32(0) == indices.dim32(0),
      "X.dim32(0) must be equal to indices.dim32(0)",
      "(",
      X.dim32(0),
      " vs. ",
      indices.dim32(0),
      ")");
  CAFFE_ENFORCE_GT(X.dim32(0), 0);

  Y->ResizeLike(X);
  const T* X_data = X.template data<T>();
  const int* indices_data = indices.template data<int>();
  T* Y_data = Y->template mutable_data<T>();

  int N = X.dim32(0);
  int K = X.numel() / N;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < N; ++i) {
    int origIdx = i * K;
    int permuteIdx = indices_data[i] * K;
    std::memcpy(Y_data + origIdx, X_data + permuteIdx, K * sizeof(T));
  }

  // Even if there is a pre-chosen quantization parameters for the output,
  // it is ignored because batch permutation output quantization should be same
  // as the input.
  PropagateOutputTensorQuantizationParams(this, 0, in_qparams_[INPUT]);

  return true;
}

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    BatchPermutation,
    DNNLOWP,
    BatchPermutationDNNLowPOp<uint8_t>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8BatchPermutation,
    DNNLOWP,
    BatchPermutationDNNLowPOp<uint8_t>);

OPERATOR_SCHEMA(Int8BatchPermutation).NumInputs(2).NumOutputs(1);

} // namespace caffe2
