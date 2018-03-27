#ifndef CAFFE2_OPERATORS_ORDER_SWITCH_OPS_H_
#define CAFFE2_OPERATORS_ORDER_SWITCH_OPS_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

// Note(Yangqing): I think it is possible to do a more general swapaxes operator
// but I am a little afraid of going down that general path. Only implementing
// the two actually needed ones here.

template <typename T, class Context>
class NHWC2NCHWOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(NHWC2NCHWOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
};

template <typename T, class Context>
class NCHW2NHWCOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(NCHW2NHWCOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ORDER_SWITCH_OPS_H_
