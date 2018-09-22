#ifndef CAFFE2_OPERATORS_RSQRT_OP_H_
#define CAFFE2_OPERATORS_RSQRT_OP_H_

#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct RsqrtFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    math::Rsqrt<T, Context>(N, X, Y, context);
    return true;
  }
};

template <class Context>
struct RsqrtGradientFunctor {
  template <typename T>
  bool Forward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& Y_dims,
      const T* dY,
      const T* Y,
      T* dX,
      Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RSQRT_OP_H_
