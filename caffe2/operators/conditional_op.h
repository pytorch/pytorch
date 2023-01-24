
#ifndef CONDITIONAL_OP_H
#define CONDITIONAL_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <class Context>
class ConditionalOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ConditionalOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;
};

} // namespace caffe2

#endif
