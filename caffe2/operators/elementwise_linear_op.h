#ifndef CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <typename T, class Context, class Engine = DefaultEngine>
class ElementwiseLinearOp final : public Operator<Context> {
  public:
   USE_SIMPLE_CTOR_DTOR(ElementwiseLinearOp);
   USE_OPERATOR_CONTEXT_FUNCTIONS;
   bool RunOnDevice() override;
};

template <typename T, class Context, class Engine = DefaultEngine>
class ElementwiseLinearGradientOp final : public Operator<Context> {
 public:
   USE_SIMPLE_CTOR_DTOR(ElementwiseLinearGradientOp);
   USE_OPERATOR_CONTEXT_FUNCTIONS;
   bool RunOnDevice() override;
};

} // namespace caffe2

#endif  // CAFFE2_OPERATORS_ELEMENTWISE_LINEAR_OP_H_
