#ifndef CAFFE2_OPERATORS_GELU_OP_H_
#define CAFFE2_OPERATORS_GELU_OP_H_

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/elementwise_ops.h"

C10_DECLARE_CAFFE2_OPERATOR(Gelu);

namespace caffe2 {

namespace gelu_utils {

constexpr float kSqrt2 = 1.4142135623730951f;
constexpr float kSqrtPi = 1.7724538509055159f;
constexpr float kFastCoeff = 0.044715f;

} // namespace gelu_utils

template <class Context>
struct GeluFunctor {
  explicit GeluFunctor(OperatorBase& op)
      : fast_gelu(op.GetSingleArgument<bool>("fast_gelu", false)) {}

  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const;

  const bool fast_gelu;
};

template <class Context>
struct GeluGradientFunctor {
  explicit GeluGradientFunctor(OperatorBase& op)
      : fast_gelu(op.GetSingleArgument<bool>("fast_gelu", false)) {}

  template <typename T>
  bool Forward(
      const std::vector<int>& dY_dims,
      const std::vector<int>& X_dims,
      const T* dY,
      const T* X,
      T* dX,
      Context* context) const;

  const bool fast_gelu;
};

template <class Context>
using GeluOp = UnaryElementwiseWithArgsOp<
    TensorTypes<float>,
    Context,
    GeluFunctor<Context>>;

template <class Context>
using GeluGradientOp = BinaryElementwiseWithArgsOp<
    TensorTypes<float>,
    Context,
    GeluGradientFunctor<Context>>;

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GELU_OP_H_
