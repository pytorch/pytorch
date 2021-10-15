#ifndef CAFFE2_OPERATORS_POOL_OP_H_
#define CAFFE2_OPERATORS_POOL_OP_H_

#include <vector>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename T, class Context, class Functor>
class PoolOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);

  template <class... Args>
  explicit PoolOp(Args&&... args)
      : ConvPoolOpBase<Context>(std::forward<Args>(args)...), functor_(*this) {
    const int kernel_size = kernel_.size();
    for (int i = 0; i < kernel_size; ++i) {
      CAFFE_ENFORCE_EQ(
          dilation_[i], 1, "Pooling op does not support dilation right now.");
    }
    if (!global_pooling_) {
      for (int i = 0; i < kernel_size; ++i) {
        CAFFE_ENFORCE(
            pads_[i] < kernel_[i] && pads_[i + kernel_size] < kernel_[i],
            "Pad should be smaller than kernel.");
      }
    }
  }

  ~PoolOp() = default;

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    ConvPoolOpBase<Context>::SetOutputSize(X, Y, C);
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    if (N == 0) {
      return true;
    }
    if (global_pooling_) {
      const int HxW = X.numel() / (N * C);
      return functor_.template GlobalPoolingForward<T, StorageOrder::NCHW>(
          N, C, HxW, X_data, Y_data, &context_);
    }
    const std::vector<int> X_HW_dims = GetDims(X);
    const std::vector<int> Y_HW_dims = GetDims(*Y);
    return functor_.template Forward<T, StorageOrder::NCHW>(
        N,
        C,
        X_HW_dims,
        Y_HW_dims,
        kernel_,
        dilation_,
        stride_,
        pads_,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = X.dim32(ndim - 1);
    ConvPoolOpBase<Context>::SetOutputSize(X, Y, C);
    const T* X_data = X.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    if (N == 0) {
      return true;
    }
    if (global_pooling_) {
      const int HxW = X.numel() / (N * C);
      return functor_.template GlobalPoolingForward<T, StorageOrder::NHWC>(
          N, C, HxW, X_data, Y_data, &context_);
    }
    const std::vector<int> X_HW_dims = GetDims(X);
    const std::vector<int> Y_HW_dims = GetDims(*Y);
    return functor_.template Forward<T, StorageOrder::NHWC>(
        N,
        C,
        X_HW_dims,
        Y_HW_dims,
        kernel_,
        dilation_,
        stride_,
        pads_,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
  }

 private:
  const Functor functor_;
};

template <typename T, class Context, class Functor>
class PoolGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  template <class... Args>
  explicit PoolGradientOp(Args&&... args)
      : ConvPoolOpBase<Context>(std::forward<Args>(args)...), functor_(*this) {}

  ~PoolGradientOp() = default;

  bool RunOnDeviceWithOrderNCHW() override {
    const auto& X = Input(0);
    const auto& Y = Input(1);
    const auto& dY = Input(2);
    auto* dX = Output(0, X.sizes(), at::dtype<T>());
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    const std::vector<int> X_HW_dims = GetDims(X);
    const std::vector<int> Y_HW_dims = GetDims(Y);
    ConvPoolOpBase<Context>::ComputePads(X_HW_dims);
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* Y_data = Y.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    if (N == 0) {
      return true;
    }
    if (global_pooling_) {
      const int HxW = X.numel() / (N * C);
      return functor_.template GlobalPoolingBackward<T, StorageOrder::NCHW>(
          N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
    }
    return functor_.template Backward<T, StorageOrder::NCHW>(
        N,
        C,
        X_HW_dims,
        Y_HW_dims,
        kernel_,
        dilation_,
        stride_,
        pads_,
        dY_data,
        X_data,
        Y_data,
        dX_data,
        &context_);
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Input(0);
    const auto& Y = Input(1);
    const auto& dY = Input(2);
    auto* dX = Output(0, X.sizes(), at::dtype<T>());
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = X.dim32(ndim - 1);
    const std::vector<int> X_HW_dims = GetDims(X);
    const std::vector<int> Y_HW_dims = GetDims(Y);
    ConvPoolOpBase<Context>::ComputePads(X_HW_dims);
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* Y_data = Y.template data<T>();
    T* dX_data = dX->template mutable_data<T>();
    if (N == 0) {
      return true;
    }
    if (global_pooling_) {
      const int HxW = X.numel() / (N * C);
      return functor_.template GlobalPoolingBackward<T, StorageOrder::NHWC>(
          N, C, HxW, dY_data, X_data, Y_data, dX_data, &context_);
    }
    return functor_.template Backward<T, StorageOrder::NHWC>(
        N,
        C,
        X_HW_dims,
        Y_HW_dims,
        kernel_,
        dilation_,
        stride_,
        pads_,
        dY_data,
        X_data,
        Y_data,
        dX_data,
        &context_);
  }

 private:
  const Functor functor_;
};

template <class Context>
struct AveragePoolFunctor {
  explicit AveragePoolFunctor(const OperatorBase& op)
      : count_include_pad(
            op.template GetSingleArgument<bool>("count_include_pad", false)) {}

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingForward(
      int N,
      int C,
      int HxW,
      const T* X,
      T* Y,
      Context* context) const;

  template <typename T, StorageOrder kOrder>
  bool Forward(
      int N,
      int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* X,
      T* Y,
      Context* context) const;

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingBackward(
      int N,
      int C,
      int HxW,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      Context* context) const;

  template <typename T, StorageOrder kOrder>
  bool Backward(
      int N,
      int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      Context* context) const;

  const bool count_include_pad;
  Tensor ones{Context::GetDeviceType()};
};

template <class Context>
struct MaxPoolFunctor {
  explicit MaxPoolFunctor(const OperatorBase& /* op */) {}

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingForward(
      int N,
      int C,
      int HxW,
      const T* X,
      T* Y,
      Context* context) const;

  template <typename T, StorageOrder kOrder>
  bool Forward(
      int N,
      int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* X,
      T* Y,
      Context* context) const;

  template <typename T, StorageOrder kOrder>
  bool GlobalPoolingBackward(
      int N,
      int C,
      int HxW,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      Context* context) const;

  template <typename T, StorageOrder kOrder>
  bool Backward(
      int N,
      int C,
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const std::vector<int>& kernel,
      const std::vector<int>& dilation,
      const std::vector<int>& stride,
      const std::vector<int>& pads,
      const T* dY,
      const T* X,
      const T* Y,
      T* dX,
      Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_POOL_OP_H_
