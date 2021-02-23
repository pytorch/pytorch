#ifndef CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_
#define CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_

#include <array>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class InstanceNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit InstanceNormOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_GE(epsilon_, 0, "Must pass a nonnegative epsilon.");
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
  }

  bool RunOnDevice() {
    const auto& X = Input(INPUT);
    const auto& gamma = Input(SCALE);
    const auto& beta = Input(BIAS);
    const int ndim = X.dim();
    const int64_t N = X.dim(0);
    const int64_t C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(ndim - 1);
    const int64_t HxW = X.numel() / (N * C);
    CAFFE_ENFORCE_EQ(gamma.numel(), C);
    CAFFE_ENFORCE_EQ(beta.numel(), C);
    auto* Y = Output(OUTPUT, X.sizes(), at::dtype<T>());
    const T* X_data = X.template data<T>();
    const T* gamma_data = gamma.template data<T>();
    const T* beta_data = beta.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    T* mean_data = nullptr;
    T* rstd_data = nullptr;
    if (OutputSize() >= 2) {
      auto* mean = Output(MEAN, {N, C}, at::dtype<T>());
      mean_data = mean->template mutable_data<T>();
    } else {
      ReinitializeTensor(
          &mean_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
      mean_data = mean_.template mutable_data<T>();
    }
    if (OutputSize() >= 3) {
      auto* rstd = Output(RSTD, {N, C}, at::dtype<T>());
      rstd_data = rstd->template mutable_data<T>();
    } else {
      ReinitializeTensor(
          &rstd_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
      rstd_data = rstd_.template mutable_data<T>();
    }
    switch (order_) {
      case StorageOrder::NCHW: {
        return RunOnDeviceWithOrderNCHW(
            N,
            C,
            HxW,
            X_data,
            gamma_data,
            beta_data,
            Y_data,
            mean_data,
            rstd_data);
      }
      case StorageOrder::NHWC: {
        return RunOnDeviceWithOrderNHWC(
            N,
            C,
            HxW,
            X_data,
            gamma_data,
            beta_data,
            Y_data,
            mean_data,
            rstd_data);
      }
      default: {
        CAFFE_THROW("Unknown storage order: ", order_);
      }
    }
  }

 private:
  bool RunOnDeviceWithOrderNCHW(
      int64_t N,
      int64_t C,
      int64_t HxW,
      const T* X,
      const T* gamma,
      const T* beta,
      T* Y,
      T* mean,
      T* rstd);

  bool RunOnDeviceWithOrderNHWC(
      int64_t N,
      int64_t C,
      int64_t HxW,
      const T* X,
      const T* gamma,
      const T* beta,
      T* Y,
      T* mean,
      T* rstd);

  const float epsilon_;
  const StorageOrder order_;

  Tensor mean_;
  Tensor rstd_;
  Tensor scale_;
  Tensor bias_;

  INPUT_TAGS(INPUT, SCALE, BIAS);
  OUTPUT_TAGS(OUTPUT, MEAN, RSTD);
};

template <typename T, class Context>
class InstanceNormGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit InstanceNormGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_GE(epsilon_, 0, "Must pass a nonnegative epsilon.");
    CAFFE_ENFORCE_NE(
        order_,
        StorageOrder::UNKNOWN,
        "order should be either \"NCHW\" or \"NHWC\".");
  }

  bool RunOnDevice() {
    const auto& X = Input(INPUT);
    const auto& gamma = Input(SCALE);
    const auto& dY = Input(OUTPUT_GRAD);
    const int ndim = X.dim();
    const int64_t N = X.dim(0);
    const int64_t C = order_ == StorageOrder::NCHW ? X.dim(1) : X.dim(ndim - 1);
    const int64_t HxW = X.numel() / (N * C);
    CAFFE_ENFORCE_EQ(gamma.numel(), C);
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* gamma_data = gamma.template data<T>();
    const T* mean_data = nullptr;
    const T* rstd_data = nullptr;
    CAFFE_ENFORCE_GE(InputSize(), 4);
    CAFFE_ENFORCE_LE(InputSize(), 6);
    if (InputSize() == 6) {
      const auto& mean = Input(MEAN);
      const auto& rstd = Input(RSTD);
      mean_data = mean.template data<T>();
      rstd_data = rstd.template data<T>();
    } else {
      ReinitializeTensor(
          &mean_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
      ReinitializeTensor(
          &rstd_, {N, C}, at::dtype<T>().device(Context::GetDeviceType()));
      ComputeMoments(
          N,
          C,
          HxW,
          X_data,
          mean_.template mutable_data<T>(),
          rstd_.template mutable_data<T>());
      mean_data = mean_.template data<T>();
      rstd_data = rstd_.template data<T>();
    }

    auto* dX = Output(INPUT_GRAD, X.sizes(), at::dtype<T>());
    auto* dgamma = Output(SCALE_GRAD, gamma.sizes(), at::dtype<T>());
    auto* dbeta = Output(BIAS_GRAD, gamma.sizes(), at::dtype<T>());
    T* dX_data = dX->template mutable_data<T>();
    T* dgamma_data = dgamma->template mutable_data<T>();
    T* dbeta_data = dbeta->template mutable_data<T>();

    switch (order_) {
      case StorageOrder::NCHW: {
        return RunOnDeviceWithOrderNCHW(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            gamma_data,
            dX_data,
            dgamma_data,
            dbeta_data);
      }
      case StorageOrder::NHWC: {
        return RunOnDeviceWithOrderNHWC(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            mean_data,
            rstd_data,
            gamma_data,
            dX_data,
            dgamma_data,
            dbeta_data);
      }
      default: {
        CAFFE_THROW("Unknown storage order: ", order_);
      }
    }
  }

 private:
  void ComputeMoments(
      int64_t N,
      int64_t C,
      int64_t HxW,
      const T* X,
      T* mean,
      T* rstd);

  bool RunOnDeviceWithOrderNCHW(
      int64_t N,
      int64_t C,
      int64_t HxW,
      const T* dY,
      const T* X,
      const T* mean,
      const T* rstd,
      const T* gamma,
      T* dX,
      T* dgamma,
      T* dbeta);

  bool RunOnDeviceWithOrderNHWC(
      int64_t N,
      int64_t C,
      int64_t HxW,
      const T* dY,
      const T* X,
      const T* mean,
      const T* rstd,
      const T* gamma,
      T* dX,
      T* dgamma,
      T* dbeta);

  const float epsilon_;
  const StorageOrder order_;

  Tensor mean_;
  Tensor rstd_;
  Tensor ds_;
  Tensor db_;
  Tensor c1_;
  Tensor c2_;
  Tensor c3_;
  Tensor ones_;

  INPUT_TAGS(INPUT, SCALE, BIAS, OUTPUT_GRAD, MEAN, RSTD);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_
