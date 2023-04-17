#ifndef CAFFE2_OPERATORS_ASYNC_BARRIER_OP_H_
#define CAFFE2_OPERATORS_ASYNC_BARRIER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class AsyncNetBarrierOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(AsyncNetBarrierOp)

  bool RunOnDevice() override {
    // This is a pretty much no-op operator, since it's only purposes is make
    // sure that async_scheduling will schedule certain operations earlier than
    // others.
    //
    // Exaple where this operator can work well - mixture of data-parallel and
    // model parallel training, where one wants to force that all copies are
    // started before data-parallel part starts.
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ASYNC_BARRIER_OP_H_
