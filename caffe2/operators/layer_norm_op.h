#ifndef CAFFE2_OPERATORS_LAYER_NORM_OP_H_
#define CAFFE2_OPERATORS_LAYER_NORM_OP_H_

#include <array>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>

C10_DECLARE_CAFFE2_OPERATOR(LayerNorm)

namespace caffe2 {

template <class Context>
class LayerNormOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template<class... Args>
  explicit LayerNormOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, 1),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f) {}

  ~LayerNormOp() {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int canonical_axis = X.canonical_axis_index(axis_);
    std::vector<int64_t> moments_dims(
        X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
    moments_dims.push_back(1);
    auto* mean = Output(1, moments_dims, at::dtype<T>());
    auto* sig = Output(2, moments_dims, at::dtype<T>());
    runLayerNorm<T>(
        X, Y, mean, sig, canonical_axis, epsilon_, &scale_, &bias_, &context_);
    return true;
  }

  template <typename T>
  static void runLayerNorm(
      const Tensor& X,
      Tensor* Y,
      Tensor* mean,
      Tensor* sig,
      int canonical_axis,
      float epsilon,
      Tensor* scale_buffer,
      Tensor* bias_buffer,
      Context* context) {
    CAFFE_ENFORCE_GE(X.dim(), 2, "LayerNorm requires input dim >= 2.");
    const int M = X.size_to_dim(canonical_axis);
    const int N = X.size_from_dim(canonical_axis);
    Y->ResizeLike(X);
    scale_buffer->Resize(M);
    bias_buffer->Resize(M);

    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    T* mean_data = mean->template mutable_data<T>();
    T* sig_data = sig->template mutable_data<T>();
    T* scale_data = scale_buffer->template mutable_data<T>();
    T* bias_data = bias_buffer->template mutable_data<T>();

    const std::array<int, 2> X_dims = {M, N};
    const std::array<int, 2> Y_dims = {M, 1};
    math::Moments<T, Context>(
        2, X_dims.data(), Y_dims.data(), X_data, mean_data, sig_data, context);
    ComputeStdDevAndFusedParams<T>(
        M,
        mean_data,
        sig_data,
        sig_data,
        scale_data,
        bias_data,
        epsilon,
        context);
    LayerNormForward<T>(M, N, X_data, scale_data, bias_data, Y_data, context);
  }

 private:
  template <typename T>
  static void ComputeStdDevAndFusedParams(
      const int N,
      const T* mean,
      const T* var,
      T* stddev,
      T* scale,
      T* bias,
      float epsilon,
      Context* context);

  template <typename T>
  static void LayerNormForward(
      const int M,
      const int N,
      const T* X,
      const T* scale,
      const T* bias,
      T* Y,
      Context* context);

  const int axis_;
  const float epsilon_;

  Tensor scale_{Context::GetDeviceType()};
  Tensor bias_{Context::GetDeviceType()};
};

template <class Context>
class LayerNormGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LayerNormGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, 1) {}

  ~LayerNormGradientOp() {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& Y = Input(1);
    const auto& mean = Input(2);
    const auto& sig = Input(3);
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
        &dY_scale_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &X_scale_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    ReinitializeTensor(
        &bias_, {M}, at::dtype<T>().device(Context::GetDeviceType()));
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* mean_data = mean.template data<T>();
    const T* sig_data = sig.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    T* ds_data = ds_.template mutable_data<T>();
    T* db_data = db_.template mutable_data<T>();
    T* dY_scale_data = dY_scale_.template mutable_data<T>();
    T* X_scale_data = X_scale_.template mutable_data<T>();
    T* bias_data = bias_.template mutable_data<T>();

    ComputeInternalGradients<T>(M, N, dY_data, X_data, ds_data, db_data);
    ComputeFusedParams<T>(
        M,
        N,
        mean_data,
        sig_data,
        ds_data,
        db_data,
        dY_scale_data,
        X_scale_data,
        bias_data);
    LayerNormBackward<T>(
        M, N, dY_scale_data, dY_data, X_scale_data, X_data, bias_data, dX_data);

    return true;
  }

 private:
  template <typename T>
  void ComputeInternalGradients(
      const int M,
      const int N,
      const T* dY,
      const T* X,
      T* ds,
      T* db);

  template <typename T>
  void ComputeFusedParams(
      const int M,
      const int N,
      const T* mean,
      const T* sig,
      const T* ds,
      const T* db,
      T* dY_scale,
      T* X_scale,
      T* bias);

  template <typename T>
  void LayerNormBackward(
      const int M,
      const int N,
      const T* dY_scale,
      const T* dY,
      const T* X_scale,
      const T* X,
      const T* bias,
      T* dX);

  const int axis_;

  Tensor ds_;
  Tensor db_;
  Tensor dY_scale_;
  Tensor X_scale_;
  Tensor bias_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LAYER_NORM_OP_H_
