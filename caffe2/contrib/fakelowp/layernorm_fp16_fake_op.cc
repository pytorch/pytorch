#include "layernorm_fp16_fake_op.h"

namespace caffe2 {

template <>
template <typename T>
void LayerNormFakeFp16Op<CPUContext>::ComputeSigmaAndFusedParams(
    const int M,
    const float eps,
    const T* mean,
    const T* var,
    T* sigma,
    T* scale,
    T* bias) {
  ConstEigenVectorArrayMap<T> var_arr(sigma, M);
  EigenVectorArrayMap<T> sigma_arr(sigma, M);
  sigma_arr = var_arr + static_cast<T>(eps);
  math::Rsqrt<T, CPUContext>(M, sigma, scale, &context_);
  math::Mul<T, CPUContext>(M, scale, sigma, sigma, &context_);
  EigenVectorArrayMap<T>(bias, M) = -ConstEigenVectorArrayMap<T>(scale, M) *
      ConstEigenVectorArrayMap<T>(mean, M);
}

template <>
template <typename T>
void LayerNormFakeFp16Op<CPUContext>::LayerNormForward(
    const int M,
    const int N,
    const T* X,
    const T* scale,
    const T* bias,
    const T* gamma,
    const T* beta,
    T* Y) {
  ConstEigenArrayMap<T> X_arr(X, N, M);
  ConstEigenVectorArrayMap<T> scale_arr(scale, M);
  ConstEigenVectorArrayMap<T> bias_arr(bias, M);
  EigenArrayMap<T> Y_arr(Y, N, M);
  if (gamma != nullptr && beta != nullptr) {
    ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
    ConstEigenVectorArrayMap<T> beta_arr(beta, N);
    Y_arr = (((X_arr.rowwise() * scale_arr.transpose()).rowwise() +
              bias_arr.transpose())
                 .colwise() *
             gamma_arr)
                .colwise() +
        beta_arr;
  } else {
    CAFFE_ENFORCE(gamma == nullptr);
    CAFFE_ENFORCE(beta == nullptr);
    Y_arr = (X_arr.rowwise() * scale_arr.transpose()).rowwise() +
        bias_arr.transpose();
  }
}

REGISTER_CPU_OPERATOR(LayerNormFakeFP16, LayerNormFakeFp16Op<CPUContext>);
OPERATOR_SCHEMA(LayerNormFakeFP16).NumInputs({1, 3}).NumOutputs(3);

} // namespace caffe2
