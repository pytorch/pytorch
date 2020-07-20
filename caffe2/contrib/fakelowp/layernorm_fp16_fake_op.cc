#include "layernorm_fp16_fake_op.h"
#include "caffe2/contrib/fakelowp/common.h"
#include "caffe2/contrib/fakelowp/fp16_fma.h"

namespace caffe2 {

template <>
void LayerNormFakeFp16Op<CPUContext>::calcY(
    const int M,
    const int N,
    const float* X,
    const float* mean,
    const float* std,
    const float* gamma,
    const float* beta,
    float* Y) {
  ConstEigenArrayMap<float> X_arr(X, N, M);
  ConstEigenVectorArrayMap<float> mean_arr(mean, M);
  ConstEigenVectorArrayMap<float> std_arr(std, M);
  EigenArrayMap<float> Y_arr(Y, N, M);

  std::vector<float> normalized(N);
  for (int i = 0; i < M; ++i) {
    float normFactor = float(1.0f / std_arr[i]);
    fp16_wrap(&normFactor);

    for (int j = 0; j < N; ++j) {
      normalized[j] = X_arr.col(i)[j] - mean[i];
    }
    fbgemm::RoundToFloat16(normalized.data(), normalized.data(), N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    for (int j = 0; j < N; ++j) {
      normalized[j] *= normFactor;
    }
    fbgemm::RoundToFloat16(normalized.data(), &Y_arr.col(i)[0], N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
  }

  if (gamma != nullptr && beta != nullptr) {
    ConstEigenVectorArrayMap<float> gamma_arr(gamma, N);
    ConstEigenVectorArrayMap<float> beta_arr(beta, N);

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
float LayerNormFakeFp16Op<CPUContext>::ReducedAdd(const std::vector<float>& vec) {
  constexpr int VEC_SIZE = 32;
  std::vector<float> v(vec.begin(), vec.end());

  for (int factor = 2; factor <=32; factor *=2) {
    int range = VEC_SIZE / factor;

    for (int i = 0; i < range; ++i) { // 16
      v[i] = v[2 * i] + v[2 * i + 1];
    }
    fbgemm::RoundToFloat16(v.data(), v.data(), range, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
  }

  return v[0];
}

template <>
void LayerNormFakeFp16Op<CPUContext>::calcMeanStd(
    const int M,
    const int N,
    const float eps,
    const float* X,
    float* mean,
    float* std) {
  ConstEigenArrayMap<float> X_arr(X, N, M);

  float sqr[M];
  float var[M];
  float inv_N_val = 1.0f / N;
  fp16_wrap(&inv_N_val);
  float tmp = 0.0f;

  constexpr int VEC_SIZE = 32;
  std::vector<float> inv_N_vec(VEC_SIZE, inv_N_val);
  std::vector<float> inv_N_prod_vec(VEC_SIZE, 0);
  std::vector<float> avgVec(VEC_SIZE, 0.0f);
  std::vector<float> sqrVec(VEC_SIZE, 0.0f);
  int numVecs = N / VEC_SIZE;
  int tailSize = N - (numVecs * VEC_SIZE);

  vector<float> X_fp16(M * N);
  fbgemm::RoundToFloat16(
      X, X_fp16.data(), M * N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

  for (int i = 0; i < M; ++i) {
    std::fill(avgVec.begin(), avgVec.end(), 0.0f);
    std::fill(sqrVec.begin(), sqrVec.end(), 0.0f);
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
      LOG_EVERY_N(WARNING, 1000) << "Variance " << var[i] << " negative, resetting to 0.";
      var[i] = 0.0;
    }

    float teps = eps;
    fp16_wrap(&teps);
    tmp = var[i] + teps;
    fp16_wrap(&tmp);
    if (tmp < 0) {
      LOG_EVERY_N(WARNING, 1000) << "Variance " << var[i] << " negative, resetting to 0.";
      tmp = 0.0;
    }
    std[i] = std::sqrt(tmp);
    fp16_wrap(&std[i]);
  }
}

REGISTER_CPU_OPERATOR(LayerNormFakeFP16NNPI, LayerNormFakeFp16Op<CPUContext>);
OPERATOR_SCHEMA(LayerNormFakeFP16NNPI).NumInputs({1, 3}).NumOutputs(3);

} // namespace caffe2
