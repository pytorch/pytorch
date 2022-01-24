#ifndef CAFFE2_OPERATORS_REDUCE_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"
#include <c10/util/irange.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace caffe2 {

template <typename InputTypes, class Context, class Reducer>
class ReduceOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit ReduceOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axes_(this->template GetRepeatedArgument<int>("axes")),
        OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true),
        reducer_{this->template GetSingleArgument<bool>("allow_broadcast_fastpath", false)} {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const int ndim = X.dim();
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      for (auto& axis : axes_) {
        axis = X.canonical_axis_index(axis);
      }
      std::sort(axes_.begin(), axes_.end());
      CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
      CAFFE_ENFORCE_LT(
          axes_.back(),
          ndim,
          "Axes ids must be smaller than the dimensions of input.");
    }
    std::vector<int64_t> output_dims;
    output_dims.reserve(ndim);
    std::size_t cur_axis = 0;
    for (const auto i : c10::irange(ndim)) {
      if (cur_axis < axes_.size() && i == axes_[cur_axis]) {
        if (keep_dims_) {
          output_dims.push_back(1);
        }
        ++cur_axis;
      } else {
        output_dims.push_back(X_dims[i]);
      }
    }
    auto* Y = Output(0, output_dims, at::dtype<T>());

    std::vector<int> Y_dims = X_dims;
    for (const int axis : axes_) {
      Y_dims[axis] = 1;
    }

    return reducer_.template Forward<T>(
        X_dims,
        Y_dims,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
  }

 private:
  std::vector<int> axes_;
  const int keep_dims_;
  const Reducer reducer_;
};

template <typename InputTypes, class Context, class Reducer>
class ReduceGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit ReduceGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axes_(this->template GetRepeatedArgument<int>("axes")),
        reducer_{this->template GetSingleArgument<bool>("allow_broadcast_fastpath", false)} {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& X = Input(1);
    const auto& Y = Input(2);

    const int ndim = X.dim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      for (auto& axis : axes_) {
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
    auto* dX = Output(0, X.sizes(), at::dtype<T>());
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
  const Reducer reducer_;
};

template <class Context>
struct MinReducer {
  explicit MinReducer(bool allow_broadcast_fastpath)
    : allow_broadcast_fastpath_(allow_broadcast_fastpath) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceMin<T, Context>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.data(),
        T(1),
        X_data,
        Y_data,
        context,
        allow_broadcast_fastpath_);
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

  const bool allow_broadcast_fastpath_;
};

template <class Context>
struct MaxReducer {
  explicit MaxReducer(bool allow_broadcast_fastpath)
    : allow_broadcast_fastpath_(allow_broadcast_fastpath) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceMax<T, Context>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.data(),
        T(1),
        X_data,
        Y_data,
        context,
        allow_broadcast_fastpath_);
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

  const bool allow_broadcast_fastpath_;
};

template <class Context>
struct SumReducer {
  explicit SumReducer(bool allow_broadcast_fastpath)
    : allow_broadcast_fastpath_(allow_broadcast_fastpath) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceSum<T, Context>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.data(),
        T(1),
        X_data,
        Y_data,
        context,
        allow_broadcast_fastpath_);
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
        context,
        allow_broadcast_fastpath_);
    return true;
  }

  const bool allow_broadcast_fastpath_;
};

template <class Context>
struct MeanReducer {
  explicit MeanReducer(bool allow_broadcast_fastpath)
    : allow_broadcast_fastpath_(allow_broadcast_fastpath) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceMean<T, Context>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.data(),
        T(1),
        X_data,
        Y_data,
        context,
        allow_broadcast_fastpath_);
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
        context,
        allow_broadcast_fastpath_);
    return true;
  }

  const bool allow_broadcast_fastpath_;
};

template <class Context>
struct L1Reducer {
  explicit L1Reducer(bool allow_broadcast_fastpath)
    : allow_broadcast_fastpath_(allow_broadcast_fastpath) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceL1<T, Context>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.data(),
        T(1),
        X_data,
        Y_data,
        context,
        allow_broadcast_fastpath_);
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

  const bool allow_broadcast_fastpath_;
};

template <class Context>
struct L2Reducer {
  explicit L2Reducer(bool allow_broadcast_fastpath)
    : allow_broadcast_fastpath_(allow_broadcast_fastpath) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& Y_dims,
      const T* X_data,
      T* Y_data,
      Context* context) const {
    math::ReduceL2<T, Context>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.data(),
        T(1),
        X_data,
        Y_data,
        context,
        allow_broadcast_fastpath_);
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

  const bool allow_broadcast_fastpath_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_OPS_H_
