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
    dims_device_.Resize(ndim);
    context_.template Copy<int, CPUContext, Context>(
        ndim, X_dims.data(), dims_device_.template mutable_data<int>());
    axes_device_.Resize(axes_.size());
    context_.template Copy<int, CPUContext, Context>(
        axes_.size(), axes_.data(), axes_device_.template mutable_data<int>());
    Y->Resize(Y_dims);
    return this->Compute(
        X.size(),
        Y->size(),
        X.template data<T>(),
        Y->template mutable_data<T>());
  }

 protected:
  virtual bool
  Compute(const int X_size, const int Y_size, const T* X_data, T* Y_data) = 0;

  std::vector<int> axes_;
  const int keep_dims_;

  Tensor<Context> dims_device_;
  Tensor<Context> axes_device_;
  Tensor<Context> buffer_;
};

template <typename T, class Context>
class ReduceSumOp final : public ReduceOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceSumOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(const int X_size, const int Y_size, const T* X_data, T* Y_data)
      override {
    math::ReduceSum<T, Context>(
        X_size,
        Y_size,
        this->dims_device_.size(),
        this->dims_device_.template data<int>(),
        this->axes_device_.size(),
        this->axes_device_.template data<int>(),
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
  bool Compute(const int X_size, const int Y_size, const T* X_data, T* Y_data)
      override {
    math::ReduceMean<T, Context>(
        X_size,
        Y_size,
        this->dims_device_.size(),
        this->dims_device_.template data<int>(),
        this->axes_device_.size(),
        this->axes_device_.template data<int>(),
        X_data,
        Y_data,
        &context_,
        &this->buffer_);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_OPS_H_
