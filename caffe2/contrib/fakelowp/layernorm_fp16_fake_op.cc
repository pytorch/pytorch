#include "layernorm_fp16_fake_op.h"
#include "caffe2/contrib/fakelowp/common.h"

namespace caffe2 {

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
        tmp = normalized * gamma_arr[j] + beta_arr[j];
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
T LayerNormFakeFp16Op<CPUContext>::ReducedAdd(const std::vector<T>& vec) {
  constexpr int VEC_SIZE = 32;
  std::vector<T> v(VEC_SIZE, T(0));
  for (int i = 0; i < VEC_SIZE; ++i) { // 32
    v[i] = vec[i];
  }
  for (int i = 0; i < VEC_SIZE / 2; ++i) { // 16
    v[i] = v[2 * i] + v[2 * i + 1];
    fp16_wrap(&v[i]);
  }
  for (int i = 0; i < VEC_SIZE / 4; ++i) { // 8
    v[i] = v[2 * i] + v[2 * i + 1];
    fp16_wrap(&v[i]);
  }
  for (int i = 0; i < VEC_SIZE / 8; ++i) { // 4
    v[i] = v[2 * i] + v[2 * i + 1];
    fp16_wrap(&v[i]);
  }
  for (int i = 0; i < VEC_SIZE / 16; ++i) { // 2
    v[i] = v[2 * i] + v[2 * i + 1];
    fp16_wrap(&v[i]);
  }
  v[0] = v[0] + v[1];
  fp16_wrap(&v[0]);
  return v[0];
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
  fp16_wrap(&inv_N_val);
  T tmp = T(0);

  const int VEC_SIZE = 32;
  std::vector<T> avgVec(VEC_SIZE, T(0));
  std::vector<T> sqrVec(VEC_SIZE, T(0));
  int numVecs = N / VEC_SIZE;
  int tailSize = N - (numVecs * VEC_SIZE);
  for (int i = 0; i < M; ++i) {
    std::fill(avgVec.begin(), avgVec.end(), T(0));
    std::fill(sqrVec.begin(), sqrVec.end(), T(0));
    for (int j = 0; j < numVecs; ++j) {
      for (int k = 0; k < VEC_SIZE; ++k) {
        avgVec[k] = X_arr.col(i)[VEC_SIZE * j + k] * inv_N_val + avgVec[k];
        fp16_wrap(&avgVec[k]);
        tmp = X_arr.col(i)[VEC_SIZE * j + k] * inv_N_val;
        fp16_wrap(&tmp);
        sqrVec[k] = tmp * X_arr.col(i)[VEC_SIZE * j + k] + sqrVec[k];
        fp16_wrap(&sqrVec[k]);
      }
    }
    for (int k = 0; k < tailSize; ++k) {
      avgVec[k] = X_arr.col(i)[VEC_SIZE * numVecs + k] * inv_N_val + avgVec[k];
      fp16_wrap(&avgVec[k]);
      tmp = X_arr.col(i)[VEC_SIZE * numVecs + k] * inv_N_val;
      fp16_wrap(&tmp);
      sqrVec[k] = tmp * X_arr.col(i)[VEC_SIZE * numVecs + k] + sqrVec[k];
      fp16_wrap(&sqrVec[k]);
    }
    mean[i] = ReducedAdd(avgVec);
    sqr[i] = ReducedAdd(sqrVec);
    // compute variance and std deviation
    var[i] = -mean[i] * mean[i] + sqr[i];
    fp16_wrap(&var[i]);
    tmp = var[i] + eps;
    fp16_wrap(&tmp);
    std[i] = std::sqrt(tmp);
    fp16_wrap(&std[i]);
  }
}

REGISTER_CPU_OPERATOR(LayerNormFakeFP16NNPI, LayerNormFakeFp16Op<CPUContext>);
OPERATOR_SCHEMA(LayerNormFakeFP16NNPI).NumInputs({1, 3}).NumOutputs(3);

} // namespace caffe2
