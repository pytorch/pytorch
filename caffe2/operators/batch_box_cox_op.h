#ifndef CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
#define CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class BatchBoxCoxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchBoxCoxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  INPUT_TAGS(DATA, LAMBDA1, LAMBDA2);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
