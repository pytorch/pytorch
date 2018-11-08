#ifndef CAFFE2_OPERATORS_REDUCE_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_OPS_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename InputTypes, class Context, class Reducer>
class ReduceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(this->template GetRepeatedArgument<int>("axes")),
        OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.dim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      for (auto& axis: axes_) {
        axis = X.canonical_axis_index(axis);
      }
      std::sort(axes_.begin(), axes_.end());
      CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
      CAFFE_ENFORCE_LT(
          axes_.back(),
          ndim,
          "Axes ids must be smaller than the dimensions of input.");
    }
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    std::vector<int> Y_dims;
    Y_dims.reserve(ndim);
    std::size_t cur_axis = 0;
    for (int i = 0; i < ndim; ++i) {
      if (cur_axis < axes_.size() && i == axes_[cur_axis]) {
        if (keep_dims_) {
          Y_dims.push_back(1);
        }
        ++cur_axis;
      } else {
        Y_dims.push_back(X_dims[i]);
      }
    }
    Y->Resize(Y_dims);
    return reducer_.template Forward<T>(
        X_dims,
        axes_,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
  }

 private:
  std::vector<int> axes_;
  const int keep_dims_;
  Reducer reducer_{};
};

template <typename InputTypes, class Context, class Reducer>
class ReduceGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(this->template GetRepeatedArgument<int>("axes")) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& X = Input(1);
    const auto& Y = Input(2);
    auto* dX = Output(0);
    const int ndim = X.dim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      for (auto& axis: axes_) {
        axis = X.canonical_axis_index(axis);
      }
      std::sort(axes_.begin(), axes_.end());
      CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
      CAFFE_ENFORCE_LT(
          axes_.back(),
          ndim,
          "Axes ids must be smaller than the dimensions of input.");
    }
    const std::vector<int> dX_dims(X.sizes().cbegin(), X.sizes().cend());
    std::vector<int> dY_dims = dX_dims;
    for (const int axis : axes_) {
      dY_dims[axis] = 1;
    }
    dX->ResizeLike(X);
    return reducer_.template Backward<T>(
        dY_dims,
        dX_dims,
        dY.template data<T>(),
        X.template data<T>(),
        Y.template data<T>(),
        dX->template mutable_data<T>(),
        &context_);
  }

 private:
  std::vector<int> axes_;
  Reducer reducer_{};
};

template <class Context>
struct MinReducer {
  template <typename T>
  bool Forward(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceMin<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        T(1),
        X_data,
        Y_data,
        context);
    return true;
  }

  template <typename T>
  bool Backward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY_data,
      const T* X_data,
      const T* Y_data,
      T* dX_data,
      Context* context) const;
};

template <class Context>
struct MaxReducer {
  template <typename T>
  bool Forward(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceMax<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        T(1),
        X_data,
        Y_data,
        context);
    return true;
  }

  template <typename T>
  bool Backward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY_data,
      const T* X_data,
      const T* Y_data,
      T* dX_data,
      Context* context) const;
};

template <class Context>
struct SumReducer {
  template <typename T>
  bool Forward(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceSum<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        T(1),
        X_data,
        Y_data,
        context);
    return true;
  }

  template <typename T>
  bool Backward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY_data,
      const T* /* X_data */,
      const T* /* Y_data */,
      T* dX_data,
      Context* context) const {
    math::Broadcast(
        dY_dims.size(),
        dY_dims.data(),
        dX_dims.size(),
        dX_dims.data(),
        T(1),
        dY_data,
        dX_data,
        context);
    return true;
  }
};

template <class Context>
struct MeanReducer {
  template <typename T>
  bool Forward(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceMean<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        T(1),
        X_data,
        Y_data,
        context);
    return true;
  }

  template <typename T>
  bool Backward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY_data,
      const T* /* X_data */,
      const T* /* Y_data */,
      T* dX_data,
      Context* context) const {
    const int dY_size = std::accumulate(
        dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
    const int dX_size = std::accumulate(
        dX_dims.cbegin(), dX_dims.cend(), 1, std::multiplies<int>());
    math::Broadcast(
        dY_dims.size(),
        dY_dims.data(),
        dX_dims.size(),
        dX_dims.data(),
        static_cast<T>(dY_size) / static_cast<T>(dX_size),
        dY_data,
        dX_data,
        context);
    return true;
  }
};

template <class Context>
struct L1Reducer {
  template <typename T>
  bool Forward(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceL1<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        T(1),
        X_data,
        Y_data,
        context);
    return true;
  }

  template <typename T>
  bool Backward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY_data,
      const T* X_data,
      const T* Y_data,
      T* dX_data,
      Context* context) const;
};

template <class Context>
struct L2Reducer {
  template <typename T>
  bool Forward(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceL2<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        T(1),
        X_data,
        Y_data,
        context);
    return true;
  }

  template <typename T>
  bool Backward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY_data,
      const T* X_data,
      const T* Y_data,
      T* dX_data,
      Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_OPS_H_
