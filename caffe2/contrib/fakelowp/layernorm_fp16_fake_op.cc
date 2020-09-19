#include <algorithm>
#include "layernorm_fp16_fake_op.h"
#include "caffe2/contrib/fakelowp/common.h"
#include "caffe2/contrib/fakelowp/fp16_fma.h"

namespace caffe2 {

void LayerNormUtils::calcY(
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
    fbgemm::RoundToFloat16(&normFactor, &normFactor, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

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

float LayerNormUtils::ReducedAdd(const std::vector<float>& vec) {
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

void LayerNormUtils::calcMeanStd(
    const int M,
    const int N,
    const float eps,
    const float* X,
    float* mean,
    float* std) {
  ConstEigenArrayMap<float> X_arr(X, N, M);

  std::vector<float> sqr(M, 0.0f);
  std::vector<float> var(M, 0.0f);
  float inv_N_val = 1.0f / N;
  fbgemm::RoundToFloat16(&inv_N_val, &inv_N_val, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);

  constexpr int VEC_SIZE = 32;
  std::vector<float> inv_N_vec(VEC_SIZE, inv_N_val);
  std::vector<float> inv_N_prod_vec(VEC_SIZE, 0);
  std::vector<float> avgVec(VEC_SIZE, 0.0f);
  std::vector<float> sqrVec(VEC_SIZE, 0.0f);
  std::vector<float> negMeanVec(M, 0.0f);
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
  }

  // // compute variance and std deviation
  std::copy(mean, mean + M, negMeanVec.begin());
  std::transform(negMeanVec.cbegin(),
      negMeanVec.cend(),
      negMeanVec.begin(),
      std::negate<float>());
  fake_fp16::fma_fp16(M, mean, negMeanVec.data(), sqr.data());
  std::copy(sqr.cbegin(), sqr.cend(), var.begin());

  float teps = eps;
  std::vector<float> tmpVec(M, 0.0f);
  fbgemm::RoundToFloat16(&teps, &teps, 1, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
  int i = 0;
  for (auto& v: var) {
    if (v < 0.0) {
      LOG_EVERY_N(WARNING, 1000) << "Variance " << v
          << " negative, resetting to 0.";
      v = 0.0;
    }
    tmpVec[i] = var[i] + teps;
    ++i;
  }
  fbgemm::RoundToFloat16(
      tmpVec.data(),
      tmpVec.data(),
      M,
      FLAGS_caffe2_fbgemm_fake_fp16_clamp);
  i = 0;
  for (auto& v: tmpVec) {
    if (v < 0) {
      LOG_EVERY_N(WARNING, 1000) << "Variance " << v
          << " negative, resetting to 0.";
      v = 0.0;
    }
    std[i] = std::sqrt(v);
    ++i;
  }
  fbgemm::RoundToFloat16(
    std,
    std,
    M,
    FLAGS_caffe2_fbgemm_fake_fp16_clamp);
}

REGISTER_CPU_OPERATOR(LayerNormFakeFP16NNPI, LayerNormFakeFp16Op<false>);
OPERATOR_SCHEMA(LayerNormFakeFP16NNPI).NumInputs({1, 3}).NumOutputs(3);

REGISTER_CPU_OPERATOR(LayerNormInt8QuantizeFakeNNPI,
                      LayerNormFakeFp16Op<true>);
OPERATOR_SCHEMA(LayerNormInt8QuantizeFakeNNPI)
    .IdenticalTypeAndShape()
    .NumInputs({1, 3})
    .NumOutputs(3);

} // namespace caffe2
