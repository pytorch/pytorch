#ifndef CAFFE_OPERATORS_MOD_OP_H_
#define CAFFE_OPERATORS_MOD_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class ModOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ModOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    divisor_ = OperatorBase::GetSingleArgument<int64_t>("divisor", 0);
    CAFFE_ENFORCE_NE(divisor_, 0, "divisor must not be 0");
    sign_follow_divisor_ =
        OperatorBase::GetSingleArgument<bool>("sign_follow_divisor", false);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  INPUT_TAGS(DATA);

 private:
  int64_t divisor_;
  bool sign_follow_divisor_;
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_MOD_OP_H_
