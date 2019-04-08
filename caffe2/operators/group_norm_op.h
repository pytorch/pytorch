#ifndef CAFFE2_OPERATORS_GROUP_NORM_OP_H_
#define CAFFE2_OPERATORS_GROUP_NORM_OP_H_

#include <array>
#include <string>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class GroupNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit GroupNormOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "group", group_, 32),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))),
        OP_SINGLE_ARG(bool, OpSchema::Arg_IsTest, is_test_, true) {
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
    if (!is_test_) {
      CAFFE_ENFORCE_EQ(OutputSize(), 3);
    }
  }

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& gamma = Input(GAMMA);
    const auto& beta = Input(BETA);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.numel() / (N * C);
    CAFFE_ENFORCE_EQ(C % group_, 0);
    CAFFE_ENFORCE_EQ(gamma.numel(), C);
    CAFFE_ENFORCE_EQ(beta.numel(), C);
    const int G = group_;
    const int K = C / G;
    auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
    T* mu_data = nullptr;
    T* rsig_data = nullptr;
    if (OutputSize() == 3) {
      auto* mu = Output(MU, {N, G}, at::dtype<T>());
      auto* rsig = Output(INV_SIGMA, {N, G}, at::dtype<T>());
      mu_data = mu->template mutable_data<T>();
      rsig_data = rsig->template mutable_data<T>();
    } else {
      ReinitializeTensor(
          &mu_, {N, G}, at::dtype<T>().device(Context::GetDeviceType()));
      ReinitializeTensor(
          &rsig_, {N, G}, at::dtype<T>().device(Context::GetDeviceType()));
      mu_data = mu_.template mutable_data<T>();
      rsig_data = rsig_.template mutable_data<T>();
    }
    if (order_ == StorageOrder::NCHW) {
      return RunOnDeviceWithOrderNCHW(
          N,
          G,
          K,
          HxW,
          X.template data<T>(),
          gamma.template data<T>(),
          beta.template data<T>(),
          Y->template mutable_data<T>(),
          mu_data,
          rsig_data);
    } else {
      return RunOnDeviceWithOrderNHWC(
          N,
          G,
          K,
          HxW,
          X.template data<T>(),
          gamma.template data<T>(),
          beta.template data<T>(),
          Y->template mutable_data<T>(),
          mu_data,
          rsig_data);
    }
  }

 private:
  bool RunOnDeviceWithOrderNCHW(
      const int N,
      const int G,
      const int K,
      const int HxW,
      const T* X,
      const T* gamma,
      const T* beta,
      T* Y,
      T* mu,
      T* rsig) {
    const int C = G * K;
    ReinitializeTensor(
        &scale_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &bias_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
    T* scale_data = scale_.template mutable_data<T>();
    T* bias_data = bias_.template mutable_data<T>();
    const std::array<int, 2> X_dims = {N * G, K * HxW};
    const std::array<int, 2> Y_dims = {N * G, 1};
    math::Moments<T, Context>(
        2, X_dims.data(), Y_dims.data(), X, mu, rsig, &context_);
    math::InvStd<T, Context>(
        N * G, static_cast<T>(epsilon_), rsig, rsig, &context_);
    ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data);
    GroupNormForwardNCHW(N, C, HxW, X, scale_data, bias_data, Y);
    return true;
  }

  bool RunOnDeviceWithOrderNHWC(
      const int N,
      const int G,
      const int K,
      const int HxW,
      const T* X,
      const T* gamma,
      const T* beta,
      T* Y,
      T* mu,
      T* rsig) {
    const int C = G * K;
    ReinitializeTensor(
        &scale_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &bias_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
    T* scale_data = scale_.template mutable_data<T>();
    T* bias_data = bias_.template mutable_data<T>();
    const std::array<int, 4> X_dims = {N, HxW, G, K};
    const std::array<int, 4> Y_dims = {N, 1, G, 1};
    math::Moments<T, Context>(
        4, X_dims.data(), Y_dims.data(), X, mu, rsig, &context_);
    math::InvStd<T, Context>(
        N * G, static_cast<T>(epsilon_), rsig, rsig, &context_);
    ComputeFusedParams(N, G, K, mu, rsig, gamma, beta, scale_data, bias_data);
    GroupNormForwardNHWC(N, C, HxW, X, scale_data, bias_data, Y);
    return true;
  }

  void ComputeFusedParams(
      int N,
      int G,
      int K,
      const T* mu,
      const T* rsig,
      const T* gamma,
      const T* beta,
      T* scale,
      T* bias);

  void GroupNormForwardNCHW(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      const T* scale,
      const T* bias,
      T* Y);

  void GroupNormForwardNHWC(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      const T* scale,
      const T* bias,
      T* Y);

  const int group_;
  const float epsilon_;
  const StorageOrder order_;
  const bool is_test_;

  Tensor mu_;
  Tensor rsig_;
  Tensor scale_;
  Tensor bias_;

  // Input: X, gamma, beta
  // Output: Y, mu, inv_sig
  INPUT_TAGS(INPUT, GAMMA, BETA);
  OUTPUT_TAGS(OUTPUT, MU, INV_SIGMA);
};

template <typename T, class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit GroupNormGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "group", group_, 32),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
  }

  bool RunOnDevice() override {
    const auto& dY = Input(OUTPUT_GRAD);
    const auto& X = Input(INPUT);
    const auto& gamma = Input(GAMMA);
    const auto& beta = Input(BETA);
    const auto& mu = Input(MU);
    const auto& rsig = Input(INV_SIGMA);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.numel() / (N * C);
    CAFFE_ENFORCE_EQ(C % group_, 0);
    CAFFE_ENFORCE_EQ(gamma.numel(), C);
    CAFFE_ENFORCE_EQ(beta.numel(), C);
    const int G = group_;
    const int K = C / G;
    auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());
    auto* dgamma = Output(GAMMA_GRAD, gamma.sizes(), at::dtype<T>());
    auto* dbeta = Output(BETA_GRAD, beta.sizes(), at::dtype<T>());
    if (order_ == StorageOrder::NCHW) {
      return RunOnDeviceWithOrderNCHW(
          N,
          G,
          K,
          HxW,
          dY.template data<T>(),
          X.template data<T>(),
          mu.template data<T>(),
          rsig.template data<T>(),
          gamma.template data<T>(),
          dX->template mutable_data<T>(),
          dgamma->template mutable_data<T>(),
          dbeta->template mutable_data<T>());
    } else {
      return RunOnDeviceWithOrderNHWC(
          N,
          G,
          K,
          HxW,
          dY.template data<T>(),
          X.template data<T>(),
          mu.template data<T>(),
          rsig.template data<T>(),
          gamma.template data<T>(),
          dX->template mutable_data<T>(),
          dgamma->template mutable_data<T>(),
          dbeta->template mutable_data<T>());
    }
  }

 protected:
  bool RunOnDeviceWithOrderNCHW(
      int N,
      int G,
      int K,
      int HxW,
      const T* dY_data,
      const T* X_data,
      const T* mu_data,
      const T* rsig_data,
      const T* gamma_data,
      T* dX_data,
      T* dgamma_data,
      T* dbeta_data);

  bool RunOnDeviceWithOrderNHWC(
      int N,
      int G,
      int K,
      int HxW,
      const T* dY_data,
      const T* X_data,
      const T* mu_data,
      const T* rsig_data,
      const T* gamma_data,
      T* dX_data,
      T* dgamma_data,
      T* dbeta_data);

  const int group_;
  const StorageOrder order_;

  Tensor ds_;
  Tensor db_;
  Tensor dY_scale_;
  Tensor X_scale_;
  Tensor bias_;
  Tensor ones_;

  // Input: dY, X, gamma, beta, mu, inv_sig
  // Output: dX, dgamma, dbeta
  INPUT_TAGS(OUTPUT_GRAD, INPUT, GAMMA, BETA, MU, INV_SIGMA);
  OUTPUT_TAGS(INPUT_GRAD, GAMMA_GRAD, BETA_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GROUP_NORM_OP_H_
