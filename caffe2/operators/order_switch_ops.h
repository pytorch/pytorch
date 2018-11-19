#ifndef CAFFE2_OPERATORS_ORDER_SWITCH_OPS_H_
#define CAFFE2_OPERATORS_ORDER_SWITCH_OPS_H_

#include <vector>

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Note(Yangqing): I think it is possible to do a more general swapaxes operator
// but I am a little afraid of going down that general path. Only implementing
// the two actually needed ones here.

template <typename T, class Context>
class NHWC2NCHWOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  USE_SIMPLE_CTOR_DTOR(NHWC2NCHWOp);

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.dim();
    CAFFE_ENFORCE_GE(ndim, 3);
    const int N = X.dim32(0);
    const int C = X.dim32(ndim - 1);
    std::vector<int> Y_dims(ndim);
    Y_dims[0] = N;
    Y_dims[1] = C;
    int HxW = 1;
    for (int i = 2; i < ndim; ++i) {
      Y_dims[i] = X.dim32(i - 1);
      HxW *= Y_dims[i];
    }
    Y->Resize(Y_dims);
    if (X.numel() <= 0) {
      return true;
    }
    math::NHWC2NCHW<T, Context>(
        N,
        C,
        HxW,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }
};

template <typename T, class Context>
class NCHW2NHWCOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  USE_SIMPLE_CTOR_DTOR(NCHW2NHWCOp);

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    const int ndim = X.dim();
    CAFFE_ENFORCE_GE(ndim, 3);
    const int N = X.dim32(0);
    const int C = X.dim32(1);
    std::vector<int> Y_dims(ndim);
    Y_dims[0] = N;
    Y_dims[ndim - 1] = C;
    int HxW = 1;
    for (int i = 1; i < ndim - 1; ++i) {
      Y_dims[i] = X.dim32(i + 1);
      HxW *= Y_dims[i];
    }
    Y->Resize(Y_dims);
    if (X.numel() <= 0) {
      return true;
    }
    math::NCHW2NHWC<T, Context>(
        N,
        C,
        HxW,
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ORDER_SWITCH_OPS_H_
