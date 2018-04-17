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

template <typename T, class Context>
class ReduceOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(OperatorBase::GetRepeatedArgument<int>("axes")),
        OP_SINGLE_ARG(bool, "keepdims", keep_dims_, 1) {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.ndim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      std::sort(axes_.begin(), axes_.end());
      CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
      CAFFE_ENFORCE_LT(
          axes_.back(),
          ndim,
          "Axes ids must be smaller than the dimensions of input.");
    }
    const std::vector<int> X_dims(X.dims().cbegin(), X.dims().cend());
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
    return this->Compute(
        X_dims, axes_, X.template data<T>(), Y->template mutable_data<T>());
  }

 protected:
  virtual bool Compute(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data) = 0;

  std::vector<int> axes_;
  const int keep_dims_;

  Tensor<Context> buffer_;
};

template <typename T, class Context>
class ReduceSumOp final : public ReduceOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceSumOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data) override {
    math::ReduceSum<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        X_data,
        Y_data,
        &context_,
        &this->buffer_);
    return true;
  }
};

template <typename T, class Context>
class ReduceMeanOp final : public ReduceOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceMeanOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const std::vector<int>& dims,
      const std::vector<int>& axes,
      const T* X_data,
      T* Y_data) override {
    math::ReduceMean<T, Context>(
        dims.size(),
        dims.data(),
        axes.size(),
        axes.data(),
        X_data,
        Y_data,
        &context_,
        &this->buffer_);
    return true;
  }
};

template <typename T, class Context>
class ReduceGradientOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceGradientOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(OperatorBase::GetRepeatedArgument<int>("axes")) {}

  bool RunOnDevice() override {
    const auto& dY = Input(0);
    const auto& X = Input(1);
    auto* dX = Output(0);
    const int ndim = X.ndim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      std::sort(axes_.begin(), axes_.end());
      CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
      CAFFE_ENFORCE_LT(
          axes_.back(),
          ndim,
          "Axes ids must be smaller than the dimensions of input.");
    }
    const std::vector<int> dX_dims(X.dims().cbegin(), X.dims().cend());
    std::vector<int> dY_dims = dX_dims;
    for (const int axis : axes_) {
      dY_dims[axis] = 1;
    }
    dX->ResizeLike(X);
    return Compute(
        dY_dims,
        dX_dims,
        dY.template data<T>(),
        dX->template mutable_data<T>());
  }

 protected:
  virtual bool Compute(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY,
      T* dX) = 0;

  std::vector<int> axes_;
};

template <typename T, class Context>
class ReduceSumGradientOp final : public ReduceGradientOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceSumGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceGradientOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY,
      T* dX) override {
    math::Broadcast(
        dY_dims.size(),
        dY_dims.data(),
        dX_dims.size(),
        dX_dims.data(),
        dY,
        dX,
        &context_);
    return true;
  }
};

template <typename T, class Context>
class ReduceMeanGradientOp final : public ReduceGradientOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceMeanGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceGradientOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY,
      T* dX) override {
    math::Broadcast(
        dY_dims.size(),
        dY_dims.data(),
        dX_dims.size(),
        dX_dims.data(),
        dY,
        dX,
        &context_);
    const int dX_size = std::accumulate(
        dX_dims.cbegin(), dX_dims.cend(), 1, std::multiplies<int>());
    int scale = 1;
    for (const int axis : this->axes_) {
      scale *= dX_dims[axis];
    }
    math::Scale<T, Context>(
        dX_size, 1.0 / static_cast<float>(scale), dX, dX, &context_);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_OPS_H_
