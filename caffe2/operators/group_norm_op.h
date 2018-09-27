#ifndef CAFFE2_OPERATORS_GROUP_NORM_OP_H_
#define CAFFE2_OPERATORS_GROUP_NORM_OP_H_

#include <string>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

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
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
  }

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& gamma = Input(GAMMA);
    const auto& beta = Input(BETA);
    const int ndim = X.ndim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.size() / (N * C);
    CAFFE_ENFORCE_EQ(C % group_, 0);
    CAFFE_ENFORCE_EQ(gamma.size(), C);
    CAFFE_ENFORCE_EQ(beta.size(), C);
    const int G = group_;
    const int D = C / G;
    auto* Y = Output(OUTPUT);
    auto* mu = Output(MU);
    auto* rsig = Output(INV_SIGMA);
    Y->ResizeLike(X);
    mu->Resize(N, G);
    rsig->Resize(N, G);
    return RunOnDeviceImpl(
        N,
        G,
        D,
        HxW,
        X.template data<T>(),
        gamma.template data<T>(),
        beta.template data<T>(),
        Y->template mutable_data<T>(),
        mu->template mutable_data<T>(),
        rsig->template mutable_data<T>());
  }

 protected:
  bool RunOnDeviceImpl(
      const int N,
      const int G,
      const int D,
      const int HxW,
      const T* X_data,
      const T* gamma_data,
      const T* beta_data,
      T* Y_data,
      T* mu_data,
      T* rsig_data);

  const int group_;
  const float epsilon_;
  const StorageOrder order_;

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
    const int ndim = X.ndim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.size() / (N * C);
    CAFFE_ENFORCE_EQ(C % group_, 0);
    CAFFE_ENFORCE_EQ(gamma.size(), C);
    CAFFE_ENFORCE_EQ(beta.size(), C);
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
