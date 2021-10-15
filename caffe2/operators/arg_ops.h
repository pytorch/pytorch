#ifndef CAFFE2_OPERATORS_ARG_OPS_H_
#define CAFFE2_OPERATORS_ARG_OPS_H_

#include <algorithm>
#include <iterator>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

namespace caffe2 {

template <class Context, class Reducer>
class ArgOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit ArgOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "axis", axis_, -1),
        OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true) {}

  bool RunOnDevice() override {
    return DispatchHelper<
        TensorTypes<std::int32_t, std::int64_t, float, double>>::
        call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);

    const int ndim = X.dim();
    if (axis_ == -1) {
      axis_ = ndim - 1;
    }
    CAFFE_ENFORCE_GE(axis_, 0);
    CAFFE_ENFORCE_LT(axis_, ndim);
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    std::vector<int64_t> Y_dims;
    Y_dims.reserve(ndim);
    int prev_size = 1;
    int next_size = 1;
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
    auto* Y = Output(0, Y_dims, at::dtype<int64_t>());
    const int n = X_dims[axis_];
    return reducer_(
        prev_size,
        next_size,
        n,
        X.template data<T>(),
        Y->template mutable_data<int64_t>(),
        &context_);
  }

 private:
  int axis_;
  const bool keep_dims_;
  Reducer reducer_{};
};

template <class Context>
struct ArgMaxReducer {
  template <typename T>
  bool operator()(
      const int prev_size,
      const int next_size,
      const int n,
      const T* X,
      int64_t* Y,
      Context* context) const;
};

template <class Context>
struct ArgMinReducer {
  template <typename T>
  bool operator()(
      const int prev_size,
      const int next_size,
      const int n,
      const T* X,
      int64_t* Y,
      Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ARG_OPS_H_
