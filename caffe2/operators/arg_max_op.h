#ifndef CAFFE2_OPERATORS_ARG_MAX_OP_H_
#define CAFFE2_OPERATORS_ARG_MAX_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class RowWiseArgMaxOp : public Operator<Context> {
 public:
  RowWiseArgMaxOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X_IN);
  OUTPUT_TAGS(ROWWISE_ARGMAX_OUT);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DISTANCE_OP_H_
