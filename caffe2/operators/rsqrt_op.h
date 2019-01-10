#ifndef CAFFE2_OPERATORS_RSQRT_OP_H_
#define CAFFE2_OPERATORS_RSQRT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct RSqrtFunctor {
  template <typename T>
  inline void operator()(const int size, const T* X, T* Y, Context* context)
      const {
    math::InvSqrt<T, Context>(size, X, Y, context);
  }
};

template <class Context>
struct RSqrtGradientFunctor {
  template <typename T>
  inline void
  Run(const int size, const T* dY, const T* Y, T* dX, Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RSQRT_OP_H_
