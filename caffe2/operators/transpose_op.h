#ifndef CAFFE2_OPERATORS_TRANSPOSE_H_
#define CAFFE2_OPERATORS_TRANSPOSE_H_

#include <algorithm>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class TransposeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  template <class... Args>
  explicit TransposeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axes_(this->template GetRepeatedArgument<int>("axes")) {
    // We will check the legality of axes_: it should be from 0 to axes_.size().
    std::vector<int> axes_sorted = axes_;
    std::sort(axes_sorted.begin(), axes_sorted.end());
    for (std::size_t i = 0; i < axes_sorted.size(); ++i) {
      if (axes_sorted[i] != i) {
        CAFFE_THROW("Axes should be a permutation of 0 to ndim.");
      }
    }
  }

  bool RunOnDevice() override {
    // Do the actual transpose, which is implemented in DoRunWithType().
    return DispatchHelper<TensorTypes<float, double, int, int64_t>>::call(
        this, Input(0));
  }

 protected:
  template <typename T>
  void TransposeImpl(const Tensor& X, Tensor* Y) {
    const int ndim = X.dim();
    if (axes_.empty()) {
      axes_.resize(ndim);
      std::iota(axes_.rbegin(), axes_.rend(), 0);
    } else {
      CAFFE_ENFORCE_EQ(ndim, axes_.size());
    }
    const std::vector<std::int64_t> X_dims = X.sizes().vec();
    std::vector<std::int64_t> Y_dims(ndim);
    for (int i = 0; i < ndim; ++i) {
      Y_dims[i] = X_dims[axes_[i]];
    }
    Y->Resize(Y_dims);
    math::Transpose<std::int64_t, T, Context>(
        X_dims.size(),
        X_dims.data(),
        axes_.data(),
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
  }

 private:
  template <typename T>
  bool DoRunWithType() {
    TransposeImpl<T>(Input(0), Output(0));
    return true;
  }

  std::vector<int> axes_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TRANSPOSE_H_
