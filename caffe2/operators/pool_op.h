#ifndef CAFFE2_OPERATORS_POOL_OP_H_
#define CAFFE2_OPERATORS_POOL_OP_H_

#include <array>
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

  PoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws), functor_(*this) {
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
    const int ndim = X.ndim();
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    ConvPoolOpBase<Context>::SetOutputSize(X, Y, C);
    const std::vector<int> X_dims = GetDims(X);
    const std::vector<int> Y_dims = GetDims(*Y);
    const int image_ndim = ndim - 2;
    switch (image_ndim) {
      case 1: {
        return functor_.template Forward<T, StorageOrder::NCHW, 1>(
            N,
            C,
            {X.dim32(2)},
            {Y->dim32(2)},
            {kernel_[0]},
            {dilation_[0]},
            {stride_[0]},
            {pads_[0], pads_[1]},
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
      }
      case 2: {
        return functor_.template Forward<T, StorageOrder::NCHW, 2>(
            N,
            C,
            {X.dim32(2), X.dim32(3)},
            {Y->dim32(2), Y->dim32(3)},
            {kernel_[0], kernel_[1]},
            {dilation_[0], dilation_[1]},
            {stride_[0], stride_[1]},
            {pads_[0], pads_[1], pads_[2], pads_[3]},
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
      }
      case 3: {
        return functor_.template Forward<T, StorageOrder::NCHW, 3>(
            N,
            C,
            {X.dim32(2), X.dim32(3), X.dim32(4)},
            {Y->dim32(2), Y->dim32(3), Y->dim32(4)},
            {kernel_[0], kernel_[1], kernel_[2]},
            {dilation_[0], dilation_[1], dilation_[2]},
            {stride_[0], stride_[1], stride_[2]},
            {pads_[0], pads_[1], pads_[2], pads_[3], pads_[4], pads_[5]},
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
      }
      default: {
        CAFFE_THROW("Unsupported pooling dim: ", image_ndim);
        return false;
      }
    }
  }

  bool RunOnDeviceWithOrderNHWC() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.ndim();
    const int N = X.dim32(0);
    const int C = X.dim32(ndim - 1);
    ConvPoolOpBase<Context>::SetOutputSize(X, Y, C);
    const int image_ndim = ndim - 2;
    switch (image_ndim) {
      case 1: {
        return functor_.template Forward<T, StorageOrder::NHWC, 1>(
            N,
            C,
            {X.dim32(1)},
            {Y->dim32(1)},
            {kernel_[0]},
            {dilation_[0]},
            {stride_[0]},
            {pads_[0], pads_[1]},
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
      }
      case 2: {
        return functor_.template Forward<T, StorageOrder::NHWC, 2>(
            N,
            C,
            {X.dim32(1), X.dim32(2)},
            {Y->dim32(1), Y->dim32(2)},
            {kernel_[0], kernel_[1]},
            {dilation_[0], dilation_[1]},
            {stride_[0], stride_[1]},
            {pads_[0], pads_[1], pads_[2], pads_[3]},
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
      }
      case 3: {
        return functor_.template Forward<T, StorageOrder::NHWC, 3>(
            N,
            C,
            {X.dim32(1), X.dim32(2), X.dim32(3)},
            {Y->dim32(1), Y->dim32(2), Y->dim32(3)},
            {kernel_[0], kernel_[1], kernel_[2]},
            {dilation_[0], dilation_[1], dilation_[2]},
            {stride_[0], stride_[1], stride_[2]},
            {pads_[0], pads_[1], pads_[2], pads_[3], pads_[4], pads_[5]},
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
      }
      default: {
        CAFFE_THROW("Unsupported pooling dim: ", image_ndim);
        return false;
      }
    }
  }

 private:
  Functor functor_;
};

template <class Context>
struct AveragePoolFunctor {
  explicit AveragePoolFunctor(const OperatorBase& op)
      : count_include_pad(
            op.GetSingleArgument<bool>("count_include_pad", false)) {}

  template <typename T, StorageOrder kOrder, int D>
  bool Forward(
      int N,
      int C,
      const std::array<int, D>& X_dims,
      const std::array<int, D>& Y_dims,
      const std::array<int, D>& kernel,
      const std::array<int, D>& dilation,
      const std::array<int, D>& stride,
      const std::array<int, 2 * D>& pads,
      const T* X,
      T* Y,
      Context* context);

  const bool count_include_pad;
};

template <class Context>
struct MaxPoolFunctor {
  explicit MaxPoolFunctor(const OperatorBase& /* op */) {}

  template <typename T, StorageOrder kOrder, int D>
  bool Forward(
      int N,
      int C,
      const std::array<int, D>& X_dims,
      const std::array<int, D>& Y_dims,
      const std::array<int, D>& kernel,
      const std::array<int, D>& dilation,
      const std::array<int, D>& stride,
      const std::array<int, 2 * D>& pads,
      const T* X,
      T* Y,
      Context* context);
};

template <typename T, class Context, class PoolType>
class PoolGradientOp final : public ConvPoolOpBase<Context> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(Context);
  PoolGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {}
  ~PoolGradientOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X, Y, dY
  // Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_POOL_OP_H_
