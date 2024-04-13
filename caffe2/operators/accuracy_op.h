#ifndef CAFFE2_OPERATORS_ACCURACY_OP_H_
#define CAFFE2_OPERATORS_ACCURACY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class AccuracyOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit AccuracyOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        top_k_(this->template GetSingleArgument<int>("top_k", 1)) {}

  bool RunOnDevice() override;

 protected:
  int top_k_;
  INPUT_TAGS(PREDICTION, LABEL);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ACCURACY_OP_H_
