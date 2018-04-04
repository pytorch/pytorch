#ifndef CAFFE2_OPERATORS_REDUCE_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

template <typename T, class Context>
class ReduceOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    axes_ = OperatorBase::GetRepeatedArgument<int>("axes");
    keepdims_ = OperatorBase::GetSingleArgument<int>("keepdims", 1);
  }

  bool RunOnDevice() override {
    int ndim = Input(0).ndim();

    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.begin(), axes_.end(), 0);
    } else {
      std::sort(axes_.begin(), axes_.end());
      CAFFE_ENFORCE(axes_.front() >= 0, "Axes ids must be non-negative.");
      CAFFE_ENFORCE(
          axes_.back() < ndim,
          "Axes ids must be smaller than the dimensions of input.");
    }

    auto& X = Input(0);
    auto* Y = Output(0);

    vector<TIndex> y_dims = X.dims();
    TIndex Y_size = X.size();
    for (TIndex id = axes_.size() - 1; id >= 0; id--) {
      TIndex reduced_axis = axes_[id];
      Y_size /= y_dims[reduced_axis];
      if (keepdims_) {
        y_dims[reduced_axis] = 1;
      } else {
        y_dims.erase(y_dims.begin() + reduced_axis);
      }
    }
    Y->Resize(y_dims);

    return this->Compute(
        X.template data<T>(),
        X.size(),
        const_cast<vector<TIndex>&>(X.dims()),
        Y->template mutable_data<T>(),
        Y_size,
        axes_,
        y_dims,
        keepdims_);
  }

 protected:
  virtual bool Compute(
      const T* X_data,
      const TIndex X_size,
      vector<TIndex>& dims,
      T* Y_data,
      const TIndex Y_size,
      vector<int>& axes,
      vector<TIndex>& Y_dims,
      int keepdims) = 0;

 private:
  std::vector<int> axes_;
  int keepdims_;
};

template <typename T, class Context>
class ReduceSumOp : public ReduceOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceSumOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const T* X_data,
      const TIndex X_size,
      vector<TIndex>& dims,
      T* Y_data,
      const TIndex Y_size,
      vector<int>& axes,
      vector<TIndex>& Y_dims,
      int keepdims) override;
};

template <typename T, class Context>
class ReduceMeanOp : public ReduceOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ReduceMeanOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceOpBase<T, Context>(operator_def, ws) {}

 protected:
  bool Compute(
      const T* X_data,
      const TIndex X_size,
      vector<TIndex>& dims,
      T* Y_data,
      const TIndex Y_size,
      vector<int>& axes,
      vector<TIndex>& Y_dims,
      int keepdims) override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_OPS_H_
