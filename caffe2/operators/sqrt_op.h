#ifndef CAFFE2_OPERATORS_SQRT_OP_H_
#define CAFFE2_OPERATORS_SQRT_OP_H_

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct SqrtFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    math::Sqrt<T, Context>(N, X, Y, context);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SQRT_OP_H_
