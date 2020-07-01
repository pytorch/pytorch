#include "layernorm_fp16_fake_op.h"
#include "caffe2/contrib/fakelowp/common.h"
#include "caffe2/contrib/fakelowp/fp16_fma.h"

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

  for (int i = 0; i < M; ++i) {
    T normFactor = T(T(1) / std_arr[i]);
    fp16_wrap(&normFactor);
    for (int j = 0; j < N; ++j) {
      T normalized = T(X_arr.col(i)[j] - mean[i]);
      fp16_wrap(&normalized);
      normalized *= normFactor;
      fp16_wrap(&normalized);
      Y_arr.col(i)[j] = normalized;
    }
  }

  if (gamma != nullptr && beta != nullptr) {
    ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
    ConstEigenVectorArrayMap<T> beta_arr(beta, N);

    for (int i = 0; i < M; ++i) {
      vector<float> res(N);
      for (int j = 0; j < N; j++) {
        res[j] = beta[j];
      }
      fake_fp16::fma_fp16(N, &Y_arr.col(i)[0], gamma, res.data());
      for (int j = 0; j < N; j++) {
        Y_arr.col(i)[j] = res[j];
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

  constexpr int VEC_SIZE = 32;
  std::vector<T> inv_N_vec(VEC_SIZE, inv_N_val);
  std::vector<T> inv_N_prod_vec(VEC_SIZE, 0);
  std::vector<T> avgVec(VEC_SIZE, T(0));
  std::vector<T> sqrVec(VEC_SIZE, T(0));
  int numVecs = N / VEC_SIZE;
  int tailSize = N - (numVecs * VEC_SIZE);

  vector<T> X_fp16(M * N);
  fbgemm::RoundToFloat16(
      X, X_fp16.data(), M * N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

  for (int i = 0; i < M; ++i) {
    std::fill(avgVec.begin(), avgVec.end(), T(0));
    std::fill(sqrVec.begin(), sqrVec.end(), T(0));
    for (int j = 0; j < numVecs; ++j) {
      fake_fp16::fma_fp16(
          VEC_SIZE,
          &X_fp16[i * N + VEC_SIZE * j],
          inv_N_vec.data(),
          avgVec.data());
      for (int k = 0; k < VEC_SIZE; k++) {
        inv_N_prod_vec[k] = X_fp16[i * N + VEC_SIZE * j + k] * inv_N_val;
      }
      fbgemm::RoundToFloat16(
          inv_N_prod_vec.data(),
          inv_N_prod_vec.data(),
          VEC_SIZE,
          FLAGS_caffe2_fbgemm_fake_fp16_clamp);

      fake_fp16::fma_fp16(
          VEC_SIZE,
          &X_fp16[i * N + VEC_SIZE * j],
          inv_N_prod_vec.data(),
          sqrVec.data());
    }

    if (tailSize > 0) {
      fake_fp16::fma_fp16(
          tailSize,
          &X_fp16[i * N + VEC_SIZE * numVecs],
          inv_N_vec.data(),
          avgVec.data());
      for (int k = 0; k < tailSize; k++) {
        inv_N_prod_vec[k] = X_fp16[i * N + VEC_SIZE * numVecs + k] * inv_N_val;
      }
      fbgemm::RoundToFloat16(
          inv_N_prod_vec.data(),
          inv_N_prod_vec.data(),
          tailSize,
          FLAGS_caffe2_fbgemm_fake_fp16_clamp);

      fake_fp16::fma_fp16(
          tailSize,
          &X_fp16[i * N + VEC_SIZE * numVecs],
          inv_N_prod_vec.data(),
          sqrVec.data());
    }

    mean[i] = ReducedAdd(avgVec);
    sqr[i] = ReducedAdd(sqrVec);
    // compute variance and std deviation

    float neg_mean = -mean[i];
    fake_fp16::fma_fp16(1, &mean[i], &neg_mean, &sqr[i]);
    var[i] = sqr[i];

    if (var[i] < 0.0) {
      LOG(WARNING) << "Variance " << var[i] << "negative, resetting to 0.";
      var[i] = 0.0;
    }

    float teps = eps;
    fp16_wrap(&teps);
    tmp = var[i] + teps;
    fp16_wrap(&tmp);
    if (tmp < 0) {
      LOG(WARNING) << "Variance " << var[i] << "negative, resetting to 0.";
      tmp = 0.0;
    }
    std[i] = std::sqrt(tmp);
    fp16_wrap(&std[i]);
  }
}

REGISTER_CPU_OPERATOR(LayerNormFakeFP16NNPI, LayerNormFakeFp16Op<CPUContext>);
OPERATOR_SCHEMA(LayerNormFakeFP16NNPI).NumInputs({1, 3}).NumOutputs(3);

} // namespace caffe2
