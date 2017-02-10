#ifndef CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_
#define CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FloatToHalfOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FloatToHalfOp);

  bool RunOnDevice() override;
};

template <class Context>
class HalfToFloatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(HalfToFloatOp);

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_HALF_FLOAT_OPS_H_
