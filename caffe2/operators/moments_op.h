#ifndef CAFFE2_OPERATORS_MOMENTS_OP_H_
#define CAFFE2_OPERATORS_MOMENTS_OP_H_

#include <algorithm>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class MomentsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  MomentsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(this->template GetRepeatedArgument<int>("axes")),
        OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true) {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* mean = Output(0);
    auto* variance = Output(1);
    const int ndim = X.dim();
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
    mean->Resize(Y_dims);
    variance->Resize(Y_dims);
    math::Moments<float, Context>(
        X_dims.size(),
        X_dims.data(),
        axes_.size(),
        axes_.data(),
        X.template data<T>(),
        mean->template mutable_data<T>(),
        variance->template mutable_data<T>(),
        &context_);
    return true;
  }

 private:
  std::vector<int> axes_;
  const int keep_dims_;
};

template <typename T, class Context>
class MomentsGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  MomentsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(this->template GetRepeatedArgument<int>("axes")) {}

  bool RunOnDevice() override {
    const auto& dmean = Input(0);
    const auto& dvariance = Input(1);
    const auto& X = Input(2);
    const auto& mean = Input(3);
    auto* dX = Output(0);
    const int ndim = X.dim();
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
    const std::vector<int> dX_dims(X.sizes().cbegin(), X.sizes().cend());
    std::vector<int> dY_dims = dX_dims;
    for (const int axis : axes_) {
      dY_dims[axis] = 1;
    }
    dX->ResizeLike(X);
    return Compute(
        dY_dims,
        dX_dims,
        dmean.template data<T>(),
        dvariance.template data<T>(),
        X.template data<T>(),
        mean.template data<T>(),
        dX->template mutable_data<T>());
  }

 private:
  bool Compute(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dmean_data,
      const T* dvariance_data,
      const T* X_data,
      const T* mean_data,
      T* dX_data);

  std::vector<int> axes_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MOMENTS_OP_H_
