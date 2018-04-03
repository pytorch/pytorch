#ifndef CAFFE2_OPERATORS_ARG_OPS_H_
#define CAFFE2_OPERATORS_ARG_OPS_H_

#include <algorithm>
#include <iterator>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

template <typename T, class Context>
class ArgOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ArgOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true) {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.ndim();
    if (axis_ == -1) {
      axis_ = ndim - 1;
    }
    CAFFE_ENFORCE_GE(axis_, 0);
    CAFFE_ENFORCE_LT(axis_, ndim);
    const std::vector<TIndex>& X_dims = X.dims();
    std::vector<TIndex> Y_dims;
    Y_dims.reserve(ndim);
    TIndex prev_size = 1;
    TIndex next_size = 1;
    for (int i = 0; i < axis_; ++i) {
      Y_dims.push_back(X_dims[i]);
      prev_size *= X_dims[i];
    }
    if (keep_dims_) {
      Y_dims.push_back(1);
    }
    for (int i = axis_ + 1; i < ndim; ++i) {
      Y_dims.push_back(X_dims[i]);
      next_size *= X_dims[i];
    }
    Y->Resize(Y_dims);
    const TIndex n = X_dims[axis_];
    return Compute(
        X.template data<T>(),
        prev_size,
        next_size,
        n,
        Y->template mutable_data<TIndex>());
  }

 protected:
  virtual bool Compute(
      const T* X,
      const TIndex prev_size,
      const TIndex next_size,
      const TIndex n,
      TIndex* Y) = 0;

 private:
  int axis_;
  const bool keep_dims_;
};

template <typename T, class Context>
class ArgMaxOp final : public ArgOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ArgMaxOp(const OperatorDef& operator_def, Workspace* ws)
      : ArgOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const T* X,
      const TIndex prev_size,
      const TIndex next_size,
      const TIndex n,
      TIndex* Y) override;
};

template <typename T, class Context>
class ArgMinOp final : public ArgOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ArgMinOp(const OperatorDef& operator_def, Workspace* ws)
      : ArgOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const T* X,
      const TIndex prev_size,
      const TIndex next_size,
      const TIndex n,
      TIndex* Y) override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ARG_OPS_H_
