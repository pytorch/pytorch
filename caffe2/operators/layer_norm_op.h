#ifndef CAFFE2_OPERATORS_LAYER_NORM_OP_H_
#define CAFFE2_OPERATORS_LAYER_NORM_OP_H_

#include <array>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(LayerNorm)

namespace caffe2 {

template <class Context>
class LayerNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit LayerNormOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
        OP_SINGLE_ARG(bool, "elementwise_affine", elementwise_affine_, false) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    CAFFE_ENFORCE_GE(X.dim(), 2, "LayerNorm requires input dim >= 2.");
    const int canonical_axis = X.canonical_axis_index(axis_);
    std::vector<int64_t> moments_dims(
        X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
    moments_dims.push_back(1);
    auto* mean = Output(1, moments_dims, at::dtype<T>());
    auto* sigma = Output(2, moments_dims, at::dtype<T>());
    const int M = X.size_to_dim(canonical_axis);
    const int N = X.size_from_dim(canonical_axis);
    Y->ResizeLike(X);
    scale_.Resize(M);
    bias_.Resize(M);
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    T* mean_data = mean->template mutable_data<T>();
    T* sigma_data = sigma->template mutable_data<T>();
    T* scale_data = scale_.template mutable_data<T>();
    T* bias_data = bias_.template mutable_data<T>();

    if (M == 0) {
      return true;
    }

    const std::array<int, 2> X_dims = {M, N};
    const std::array<int, 2> Y_dims = {M, 1};
    math::Moments<T, Context>(
        2,
        X_dims.data(),
        Y_dims.data(),
        X_data,
        mean_data,
        sigma_data,
        &context_);
    ComputeSigmaAndFusedParams<T>(
        M, epsilon_, mean_data, sigma_data, sigma_data, scale_data, bias_data);
    const T* gamma_data = nullptr;
    const T* beta_data = nullptr;
    if (elementwise_affine_) {
      CAFFE_ENFORCE_EQ(InputSize(), 3);
      const auto& gamma = Input(1);
      const auto& beta = Input(2);
      CAFFE_ENFORCE_EQ(gamma.numel(), N);
      CAFFE_ENFORCE_EQ(beta.numel(), N);
      gamma_data = gamma.template data<T>();
      beta_data = beta.template data<T>();
    }
    LayerNormForward<T>(
        M, N, X_data, scale_data, bias_data, gamma_data, beta_data, Y_data);
    return true;
  }

 private:
  template <typename T>
  void ComputeSigmaAndFusedParams(
      const int N,
      const float eps,
      const T* mean,
      const T* var,
      T* stddev,
      T* scale,
      T* bias);

  template <typename T>
  void LayerNormForward(
      const int M,
      const int N,
      const T* X,
      const T* scale,
      const T* bias,
      const T* gamma,
      const T* beta,
      T* Y);

  const int axis_;
  const float epsilon_;
  const bool elementwise_affine_;

  Tensor scale_{Context::GetDeviceType()};
  Tensor bias_{Context::GetDeviceType()};
};

template <class Context>
class LayerNormGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit LayerNormGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1),
        OP_SINGLE_ARG(bool, "elementwise_affine", elementwise_affine_, false) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& Y = Input(1);
    const auto& mean = Input(2);
    const auto& sigma = Input(3);
    const auto& X = Input(4);

    const int canonical_axis = X.canonical_axis_index(axis_);
    const int M = X.size_to_dim(canonical_axis);
    const int N = X.size_from_dim(canonical_axis);

    auto* dX = Output(0, X.sizes(), at::dtype<T>());
    ReinitializeTensor(
        &ds_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &db_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &rstd_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &X_scale_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &bias_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* mean_data = mean.template data<T>();
    const T* sigma_data = sigma.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    T* ds_data = ds_.template mutable_data<T>();
    T* db_data = db_.template mutable_data<T>();
    T* rstd_data = rstd_.template mutable_data<T>();
    T* X_scale_data = X_scale_.template mutable_data<T>();
    T* bias_data = bias_.template mutable_data<T>();

    const T* gamma_data = nullptr;
    T* dgamma_data = nullptr;
    T* dbeta_data = nullptr;
    T* g_scale_data = nullptr;
    if (elementwise_affine_) {
      const auto& gamma = Input(5);
      auto* dgamma = Output(1, gamma.sizes(), at::dtype<T>());
      auto* dbeta = Output(2, gamma.sizes(), at::dtype<T>());
      ReinitializeTensor(
          &g_scale_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
      gamma_data = gamma.template data<T>();
      dgamma_data = dgamma->template mutable_data<T>();
      dbeta_data = dbeta->template mutable_data<T>();
      g_scale_data = g_scale_.template mutable_data<T>();
    }

    if (M == 0) {
      if (N > 0 && dgamma_data != nullptr) {
        math::Set<T, Context>(N, T(0), dgamma_data, &context_);
      }
      if (N > 0 && dbeta_data != nullptr) {
        math::Set<T, Context>(N, T(0), dbeta_data, &context_);
      }
      return true;
    }

    ComputeInternalGradients<T>(
        M, N, dY_data, X_data, gamma_data, dX_data, ds_data, db_data);
    ComputeFusedParams<T>(
        M,
        N,
        mean_data,
        sigma_data,
        ds_data,
        db_data,
        rstd_data,
        X_scale_data,
        bias_data,
        g_scale_data);
    if (elementwise_affine_) {
      GammaBetaBackward<T>(
          M,
          N,
          dX_data,
          dY_data,
          rstd_data,
          g_scale_data,
          dgamma_data,
          dbeta_data);
    }
    LayerNormBackward<T>(
        M,
        N,
        dY_data,
        X_data,
        gamma_data,
        rstd_data,
        X_scale_data,
        bias_data,
        dX_data);

    return true;
  }

 private:
  template <typename T>
  void ComputeInternalGradients(
      const int M,
      const int N,
      const T* dY,
      const T* X,
      const T* gamma,
      T* dYxX,
      T* ds,
      T* db);

  template <typename T>
  void ComputeFusedParams(
      const int M,
      const int N,
      const T* mean,
      const T* sigma,
      const T* ds,
      const T* db,
      T* rstd,
      T* X_scale,
      T* bias,
      T* g_scale);

  template <typename T>
  void LayerNormBackward(
      const int M,
      const int N,
      const T* dY,
      const T* X,
      const T* gamma,
      const T* dY_scale,
      const T* X_scale,
      const T* bias,
      T* dX);

  template <typename T>
  void GammaBetaBackward(
      const int M,
      const int N,
      const T* dYxX,
      const T* dY,
      const T* rstd,
      const T* g_scale,
      T* dgamma,
      T* dbeta);

  const int axis_;
  const bool elementwise_affine_;

  Tensor ds_;
  Tensor db_;
  Tensor rstd_;
  Tensor X_scale_;
  Tensor bias_;
  Tensor g_scale_;
  Tensor ones_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LAYER_NORM_OP_H_
