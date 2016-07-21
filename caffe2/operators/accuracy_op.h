#ifndef CAFFE2_OPERATORS_ACCURACY_OP_H_
#define CAFFE2_OPERATORS_ACCURACY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class AccuracyOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(AccuracyOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(PREDICTION, LABEL);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ACCURACY_OP_H_
