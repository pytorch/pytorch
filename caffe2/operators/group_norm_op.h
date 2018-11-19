#ifndef CAFFE2_OPERATORS_GROUP_NORM_OP_H_
#define CAFFE2_OPERATORS_GROUP_NORM_OP_H_

#include <array>
#include <string>
#include <vector>

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class GroupNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  GroupNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
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
    const int D = C / G;
    auto* Y = Output(OUTPUT);
    Y->ResizeLike(X);
    T* mu_data = nullptr;
    T* rsig_data = nullptr;
    if (OutputSize() == 3) {
      auto* mu = Output(MU);
      auto* rsig = Output(INV_SIGMA);
      mu->Resize(N, G);
      rsig->Resize(N, G);
      mu_data = mu->template mutable_data<T>();
      rsig_data = rsig->template mutable_data<T>();
    } else {
      mu_.Resize(N, G);
      rsig_.Resize(N, G);
      mu_data = mu_.template mutable_data<T>();
      rsig_data = rsig_.template mutable_data<T>();
    }
    return RunOnDeviceImpl(
        N,
        G,
        D,
        HxW,
        X.template data<T>(),
        gamma.template data<T>(),
        beta.template data<T>(),
        Y->template mutable_data<T>(),
        mu_data,
        rsig_data);
  }

 protected:
  bool RunOnDeviceImpl(
      const int N,
      const int G,
      const int D,
      const int HxW,
      const T* X,
      const T* gamma,
      const T* beta,
      T* Y,
      T* mu,
      T* rsig) {
    const int C = G * D;
    scale_.Resize(N, C);
    bias_.Resize(N, C);
    T* scale_data = scale_.template mutable_data<T>();
    T* bias_data = bias_.template mutable_data<T>();
    if (order_ == StorageOrder::NCHW) {
      const std::array<int, 2> dims = {N * G, D * HxW};
      const int axis = 1;
      math::Moments<T, Context>(
          2, dims.data(), 1, &axis, X, mu, rsig, &context_);
      math::InvStd<T, Context>(
          N * G, static_cast<T>(epsilon_), rsig, rsig, &context_);
      ComputeFusedParams(N, G, D, mu, rsig, gamma, beta, scale_data, bias_data);
      GroupNormForwardNCHW(N, C, HxW, X, scale_data, bias_data, Y);
    } else {
      const std::array<int, 4> dims = {N, HxW, G, D};
      const std::array<int, 2> axes = {1, 3};
      math::Moments<T, Context>(
          4, dims.data(), 2, axes.data(), X, mu, rsig, &context_);
      math::InvStd<T, Context>(
          N * G, static_cast<T>(epsilon_), rsig, rsig, &context_);
      ComputeFusedParams(N, G, D, mu, rsig, gamma, beta, scale_data, bias_data);
      GroupNormForwardNHWC(N, C, HxW, X, scale_data, bias_data, Y);
    }
    return true;
  }

  void ComputeFusedParams(
      const int N,
      const int G,
      const int D,
      const T* mu,
      const T* rsig,
      const T* gamma,
      const T* beta,
      T* scale,
      T* bias) {
    const int C = G * D;
    ConstEigenArrayMap<float> gamma_arr(gamma, D, G);
    ConstEigenArrayMap<float> beta_arr(beta, D, G);
    for (int i = 0; i < N; ++i) {
      EigenArrayMap<T> scale_arr(scale + i * C, D, G);
      scale_arr = gamma_arr.rowwise() *
          ConstEigenVectorArrayMap<T>(rsig + i * G, G).transpose();
      EigenArrayMap<T>(bias + i * C, D, G) = beta_arr -
          scale_arr.rowwise() *
              ConstEigenVectorArrayMap<T>(mu + i * G, G).transpose();
    }
  }

  void GroupNormForwardNCHW(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      const T* scale,
      const T* bias,
      T* Y) {
    EigenArrayMap<float>(Y, HxW, N * C) =
        (ConstEigenArrayMap<float>(X, HxW, N * C).rowwise() *
         ConstEigenVectorArrayMap<float>(scale, N * C).transpose())
            .rowwise() +
        ConstEigenVectorArrayMap<float>(bias, N * C).transpose();
  }

  void GroupNormForwardNHWC(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      const T* scale,
      const T* bias,
      T* Y) {
    const int stride = HxW * C;
    for (int i = 0; i < N; ++i) {
      EigenArrayMap<float>(Y + i * stride, C, HxW) =
          (ConstEigenArrayMap<float>(X + i * stride, C, HxW).colwise() *
           ConstEigenVectorArrayMap<float>(scale + i * C, C))
              .colwise() +
          ConstEigenVectorArrayMap<float>(bias + i * C, C);
    }
  }

  const int group_;
  const float epsilon_;
  const StorageOrder order_;
  const bool is_test_;

  Tensor mu_{Context::GetDeviceType()};
  Tensor rsig_{Context::GetDeviceType()};
  Tensor scale_{Context::GetDeviceType()};
  Tensor bias_{Context::GetDeviceType()};

  // Input: X, gamma, beta
  // Output: Y, mu, inv_sig
  INPUT_TAGS(INPUT, GAMMA, BETA);
  OUTPUT_TAGS(OUTPUT, MU, INV_SIGMA);
};

template <typename T, class Context>
class GroupNormGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  GroupNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
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
    const int D = C / G;
    auto* dX = Output(INPUT_GRAD);
    auto* dgamma = Output(GAMMA_GRAD);
    auto* dbeta = Output(BETA_GRAD);
    dX->ResizeLike(X);
    dgamma->ResizeLike(gamma);
    dbeta->ResizeLike(beta);
    return RunOnDeviceImpl(
        N,
        G,
        D,
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

 protected:
  bool RunOnDeviceImpl(
      const int N,
      const int G,
      const int D,
      const int HxW,
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

  Tensor ds_{Context::GetDeviceType()};
  Tensor db_{Context::GetDeviceType()};

  // Input: dY, X, gamma, beta, mu, inv_sig
  // Output: dX, dgamma, dbeta
  INPUT_TAGS(OUTPUT_GRAD, INPUT, GAMMA, BETA, MU, INV_SIGMA);
  OUTPUT_TAGS(INPUT_GRAD, GAMMA_GRAD, BETA_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GROUP_NORM_OP_H_
