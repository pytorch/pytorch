#ifndef CAFFE2_OPERATORS_RECIPROCAL_OP_H_
#define CAFFE2_OPERATORS_RECIPROCAL_OP_H_

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct ReciprocalFunctor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    math::Inv(N, X, Y, context);
    return true;
  }
};

template <class Context>
struct ReciprocalGradientFunctor {
  template <typename T>
  bool Forward(
      const std::vector<int>& Y_dims,
      const std::vector<int>& dY_dims,
      const T* Y,
      const T* dY,
      T* dX,
      Context* context) const;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECIPROCAL_OP_H_
