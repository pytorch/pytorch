#include "layernorm_fp16_fake_op.h"

namespace caffe2 {

template <>
template <typename T>
void LayerNormFakeFp16Op<CPUContext>::fp16_wrap(T* tmp) {
  fbgemm::RoundToFloat16(tmp, tmp, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
}

template <>
template <typename T>
void LayerNormFakeFp16Op<CPUContext>::calcY(
    const int M,
    const int N,
    const T* X,
    const T* mean,
    const T* std,
    const T* gamma,
    const T* beta,
    T* Y) {
  ConstEigenArrayMap<T> X_arr(X, N, M);
  ConstEigenVectorArrayMap<T> mean_arr(mean, M);
  ConstEigenVectorArrayMap<T> std_arr(std, M);
  EigenArrayMap<T> Y_arr(Y, N, M);
  T tmp = T(0);

  if (gamma != nullptr && beta != nullptr) {
    ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
    ConstEigenVectorArrayMap<T> beta_arr(beta, N);

    for (int i = 0; i < M; ++i) {
      T normFactor = T(T(1) / std_arr[i]);
      fp16_wrap(&normFactor);
      for (int j = 0; j < N; ++j) {
        tmp = T(X_arr.col(i)[j] - mean[i]);
        fp16_wrap(&tmp);
        T normalized = tmp * normFactor;
        fp16_wrap(&normalized);
        tmp = normalized * gamma_arr[j];
        fp16_wrap(&tmp);
        tmp = tmp + beta_arr[j];
        fp16_wrap(&tmp);
        Y_arr.col(i)[j] = tmp;
      }
    }
  } else {
    for (int i = 0; i < M; ++i) {
      T normFactor = T(T(1) / std_arr[i]);
      fp16_wrap(&normFactor);
      for (int j = 0; j < N; ++j) {
        tmp = T(X_arr.col(i)[j] - mean[i]);
        fp16_wrap(&tmp);
        tmp *= normFactor;
        fp16_wrap(&tmp);
        Y_arr.col(i)[j] = tmp;
      }
    }
  }
}

template <>
template <typename T>
void LayerNormFakeFp16Op<CPUContext>::calcMeanStd(
    const int M,
    const int N,
    const float eps,
    const T* X,
    T* mean,
    T* std) {
  ConstEigenArrayMap<T> X_arr(X, N, M);

  T sqr[M];
  T var[M];
  T inv_N_val = T(1) / N;
  T tmp = T(0);

  for (int i = 0; i < M; ++i) {
    mean[i] = T(0);
    sqr[i] = T(0);
    var[i] = T(0);
    for (int j = 0; j < N; ++j) {
      tmp = T(X_arr.col(i)[j] * inv_N_val);
      fp16_wrap(&tmp);
      mean[i] += tmp;
      fp16_wrap(&mean[i]);
      tmp *= X_arr.col(i)[j];
      fp16_wrap(&tmp);
      sqr[i] += tmp;
      fp16_wrap(&sqr[i]);
    }
    tmp = mean[i] * mean[i];
    fp16_wrap(&tmp);
    var[i] = sqr[i] - tmp;
    fp16_wrap(&var[i]);
    std[i] = std::sqrt(var[i] + eps);
    fp16_wrap(&std[i]);
  }
}

REGISTER_CPU_OPERATOR(LayerNormFakeFP16, LayerNormFakeFp16Op<CPUContext>);
OPERATOR_SCHEMA(LayerNormFakeFP16).NumInputs({1, 3}).NumOutputs(3);

} // namespace caffe2
