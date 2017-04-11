#ifndef CAFFE_OPERATORS_ONE_HOT_OPS_H_
#define CAFFE_OPERATORS_ONE_HOT_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class BatchOneHotOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BatchOneHotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(X));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  INPUT_TAGS(X, LENS, VALS);
  OUTPUT_TAGS(ONE_HOT);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_ONE_HOT_OPS_H_
