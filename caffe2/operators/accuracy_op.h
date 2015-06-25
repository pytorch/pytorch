#ifndef CAFFE2_OPERATORS_ACCURACY_OP_H_
#define CAFFE2_OPERATORS_ACCURACY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename dtype, class DeviceContext>
class AccuracyOp final : public Operator<dtype, DeviceContext> {
 public:
  USE_SIMPLE_CTOR_DTOR(AccuracyOp);
  USE_OPERATOR_BASE_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  INPUT_OUTPUT_STATS(2, 2, 1, 1);
  INPUT_TAGS(PREDICTION, LABEL);
  DISABLE_COPY_AND_ASSIGN(AccuracyOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ACCURACY_OP_H_
