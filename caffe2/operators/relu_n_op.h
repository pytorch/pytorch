#ifndef CAFFE2_OPERATORS_RELU_N_OP_H_
#define CAFFE2_OPERATORS_RELU_N_OP_H_

#include <vector>

#include "caffe2/operators/elementwise_ops.h"

namespace caffe2 {

template <class Context>
struct ReluNFunctor {
  explicit ReluNFunctor(OperatorBase& op)
      : n(op.GetSingleArgument<float>("n", 6.0f)) {
    CAFFE_ENFORCE_GT(n, 0, "n should be greater than 0");
  }

  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const;

  const float n;
};

template <class Context>
struct ReluNGradientFunctor {
  explicit ReluNGradientFunctor(OperatorBase& op)
      : n(op.GetSingleArgument<float>("n", 6.0f)) {
    CAFFE_ENFORCE_GT(n, 0, "n should be greater than 0");
  }

  template <typename T>
  bool Forward(
      const std::vector<int>& Y_dims,
      const std::vector<int>& dY_dims,
      const T* Y,
      const T* dY,
      T* dX,
      Context* context) const;

  const float n;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RELU_N_OP_H_
