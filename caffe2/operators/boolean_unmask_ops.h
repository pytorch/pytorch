#ifndef BOOLEAN_UNMASK_OPS_H
#define BOOLEAN_UNMASK_OPS_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class BooleanUnmaskOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(BooleanUnmaskOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif
