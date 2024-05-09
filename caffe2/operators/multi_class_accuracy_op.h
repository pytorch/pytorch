#ifndef CAFFE2_OPERATORS_MULTI_CLASS_ACCURACY_OP_H_
#define CAFFE2_OPERATORS_MULTI_CLASS_ACCURACY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class MultiClassAccuracyOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(MultiClassAccuracyOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(PREDICTION, LABEL);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MULTI_CLASS_ACCURACY_OP_H_
