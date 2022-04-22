#ifndef CAFFE2_OPERATORS_RMS_NORM_OP_H_
#define CAFFE2_OPERATORS_RMS_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// RMSNorm op.
// https://openreview.net/pdf?id=SygkZ3MTJE

template <class Context>
class RMSNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit RMSNormOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1),
        OP_SINGLE_ARG(float, "eps", eps_, 0.0f) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType();

 private:
  const int axis_;
  const float eps_;
};

template <class Context>
class RMSNormGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit RMSNormGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& X = Input(1);
    const auto& gamma = Input(2);
    const auto& rrms = Input(3);
    const int canonical_axis = X.canonical_axis_index(axis_);
    const int64_t M = X.size_to_dim(canonical_axis);
    const int64_t N = X.size_from_dim(canonical_axis);
    auto* dX = Output(0, X.sizes(), at::dtype<T>());
    auto* dgamma = Output(1, gamma.sizes(), at::dtype<T>());
    auto* dbeta = Output(2, gamma.sizes(), at::dtype<T>());
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* gamma_data = gamma.template data<T>();
    const T* rrms_data = rrms.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    T* dgamma_data = dgamma->template mutable_data<T>();
    T* dbeta_data = dbeta->template mutable_data<T>();

    if (M == 0) {
      math::Set<T, Context>(N, T(0), dgamma_data, &context_);
      math::Set<T, Context>(N, T(0), dbeta_data, &context_);
      return true;
    }

    RMSNormBackward<T>(M, N, dY_data, X_data, gamma_data, rrms_data, dX_data);
    GammaBetaBackward<T>(
        M, N, dY_data, X_data, rrms_data, dgamma_data, dbeta_data);

    return true;
  }

 private:
  template <typename T>
  void RMSNormBackward(
      int64_t M,
      int64_t N,
      const T* dY,
      const T* X,
      const T* gamma,
      const T* rrms,
      T* dX);

  template <typename T>
  void GammaBetaBackward(
      int64_t M,
      int64_t N,
      const T* dY,
      const T* X,
      const T* rrms,
      T* dgamma,
      T* dbeta);

  const int axis_;
  Tensor c2_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RMS_NORM_OP_H_
