#ifndef CAFFE2_OPERATORS_HARD_SIGMOID_H_
#define CAFFE2_OPERATORS_HARD_SIGMOID_H_

#include <vector>

#include "caffe2/operators/elementwise_ops.h"

namespace caffe2 {

template <class Context>
struct HardSigmoidFunctor {
  explicit HardSigmoidFunctor(OperatorBase& op)
      : alpha(op.GetSingleArgument<float>("alpha", 0.2f)),
        beta(op.GetSingleArgument<float>("beta", 0.5f)) {}

  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const;

  const float alpha, beta;
};

template <class Context>
struct HardSigmoidGradientFunctor {
  explicit HardSigmoidGradientFunctor(OperatorBase& op)
      : alpha(op.GetSingleArgument<float>("alpha", 0.2f)) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& Y_dims,
      const std::vector<int>& dY_dims,
      const T* Y,
      const T* dY,
      T* dX,
      Context* context) const;

  const float alpha;
};

} // namespace caffe2

#endif // CAFFE2CAFFE2_OPERATORS_HARD_SIGMOID_H_
