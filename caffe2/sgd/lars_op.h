#ifndef CAFFE2_OPERATORS_LARS_OP_H_
#define CAFFE2_OPERATORS_LARS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LarsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LarsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        offset_(this->template GetSingleArgument<float>("offset", 0.5)),
        lr_min_(this->template GetSingleArgument<float>("lr_min", 0.02)) {}

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& dX = Input(1);
    CAFFE_ENFORCE(
        dX.numel() == X.numel(), "Gradient size doesn't match parameter size.");
    CAFFE_ENFORCE_GE(offset_, 0);
    CAFFE_ENFORCE_GE(lr_min_, 0);

    auto& wd = Input(2);
    auto& trust = Input(3);
    auto& lr_max = Input(4);

    auto* lr_rescaled = Output(0, vector<int64_t>{1}, at::dtype<T>());

    ReinitializeTensor(&X_norm_tensor_, {1}, at::dtype<T>().device(Context::GetDeviceType()));
    T* X_norm_ = X_norm_tensor_.template mutable_data<T>();

    ReinitializeTensor(&dX_norm_tensor_, {1}, at::dtype<T>().device(Context::GetDeviceType()));
    T* dX_norm_ = dX_norm_tensor_.template mutable_data<T>();

    ComputeNorms(
        dX.numel(),
        X.template data<T>(),
        dX.template data<T>(),
        X_norm_,
        dX_norm_);

    ComputeLearningRate(
        wd.template data<T>(),
        trust.template data<T>(),
        lr_max.template data<T>(),
        offset_,
        lr_min_,
        X_norm_,
        dX_norm_,
        lr_rescaled->template mutable_data<T>());

    return true;
  }

 private:
  // Compute the l2 norm of X_data and dX_data
  void ComputeNorms(
      int64_t N,
      const T* X_data,
      const T* dX_data,
      T* X_norm,
      T* dX_norm) {
    math::SumSqr(N, X_data, X_norm, &context_);
    math::Sqrt(1, X_norm, X_norm, &context_);
    math::SumSqr(N, dX_data, dX_norm, &context_);
    math::Sqrt(1, dX_norm, dX_norm, &context_);
  }
  // Compute the learning rate and apply clipping
  void ComputeLearningRate(
      const T* wd,
      const T* trust,
      const T* lr_max,
      T offset,
      T lr_min,
      T* X_norm,
      T* dX_norm,
      T* lr_rescaled);

  T offset_;
  T lr_min_;

  Tensor X_norm_tensor_;
  Tensor dX_norm_tensor_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LARS_OP_H_
