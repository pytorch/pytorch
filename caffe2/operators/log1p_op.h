#ifndef CAFFE2_OPERATORS_LOG1P_OP_H_
#define CAFFE2_OPERATORS_LOG1P_OP_H_

#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct Log1pFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    math::Log1p(N, X, Y, context);
    return true;
  }
};

template <class Context>
struct Log1pGradientFunctor {
  template <typename T>
  bool Forward(
      const std::vector<int>& X_dims,
      const std::vector<int>& dY_dims,
      const T* X,
      const T* dY,
      T* dX,
      Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOG1P_OP_H_
