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
class InstanceNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit InstanceNormGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        epsilon_(this->template GetSingleArgument<T>("epsilon", 1e-5f)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(epsilon_ >= 0, "Must pass a nonnegative epsilon.");
  }
  ~InstanceNormGradientOp() {}

  bool RunOnDevice() {
    switch (order_) {
      case StorageOrder::NHWC:
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        return RunOnDeviceWithOrderNCHW();
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }
  }

  bool RunOnDeviceWithOrderNHWC();
  bool RunOnDeviceWithOrderNCHW();

 protected:
  // parameters
  T epsilon_;
  StorageOrder order_;

  // temp results that could get passed through to this gradient, but if not,
  // are stored here
  Tensor mean_;
  Tensor inv_stdev_;

  INPUT_TAGS(INPUT, SCALE, BIAS, OUTPUT_GRAD, MEAN, INV_STDEV);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INSTANCE_NORM_OP_H_
