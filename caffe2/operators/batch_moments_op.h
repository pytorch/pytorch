#ifndef CAFFE2_OPERATORS_BATCH_MOMENTS_OP_H_
#define CAFFE2_OPERATORS_BATCH_MOMENTS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class BatchMomentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BatchMomentsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* mu = Output(0);
    auto* var = Output(1);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.numel() / (N * C);
    mu->Resize(C);
    var->Resize(C);
    const T* X_data = X.template data<T>();
    T* mu_data = mu->template mutable_data<T>();
    T* var_data = var->template mutable_data<T>();
    return order_ == StorageOrder::NCHW
        ? ComputeBatchMomentsNCHW(N, C, HxW, X_data, mu_data, var_data)
        : ComputeBatchMomentsNHWC(N, C, HxW, X_data, mu_data, var_data);
  }

 private:
  bool ComputeBatchMomentsNCHW(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      T* mu,
      T* var);

  bool ComputeBatchMomentsNHWC(
      const int N,
      const int C,
      const int HxW,
      const T* X,
      T* mu,
      T* var);

  const StorageOrder order_;
};

template <typename T, class Context>
class BatchMomentsGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BatchMomentsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  bool RunOnDevice() override {
    const auto& dmu = Input(0);
    const auto& dvar = Input(1);
    const auto& X = Input(2);
    auto* dX = Output(0);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.numel() / (N * C);
    dX->ResizeLike(X);
    const T* dmu_data = dmu.template data<T>();
    const T* dvar_data = dvar.template data<T>();
    const T* X_data = X.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    return order_ == StorageOrder::NCHW
        ? ComputeBatchMomentsGradientNCHW(
              N, C, HxW, dmu_data, dvar_data, X_data, dX_data)
        : ComputeBatchMomentsGradientNHWC(
              N, C, HxW, dmu_data, dvar_data, X_data, dX_data);
  }

 private:
  bool ComputeBatchMomentsGradientNCHW(
      const int N,
      const int C,
      const int HxW,
      const T* dmu,
      const T* dvar,
      const T* X,
      T* dX);

  bool ComputeBatchMomentsGradientNHWC(
      const int N,
      const int C,
      const int HxW,
      const T* dmu,
      const T* dvar,
      const T* X,
      T* dX);

  const StorageOrder order_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_MOMENTS_OP_H_
