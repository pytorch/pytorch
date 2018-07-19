#ifndef CAFFE2_OPERATORS_ACCURACY_OP_H_
#define CAFFE2_OPERATORS_ACCURACY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class AccuracyOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AccuracyOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        top_k_(OperatorBase::GetSingleArgument<int>("top_k", 1)) {}
        
  bool RunOnDevice() override;

 protected:
  int top_k_; 
  INPUT_TAGS(PREDICTION, LABEL);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ACCURACY_OP_H_
