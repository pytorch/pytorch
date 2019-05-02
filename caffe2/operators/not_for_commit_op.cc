#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

class NotForCommitOp final : public Operator<CPUContext> {
 public:
  NotForCommitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    return true;
  }
};

OPERATOR_SCHEMA(NotForCommit)
    .NumInputs(0)
    .NumOutputs(0)
    .SetDoc(
        R"DOC(Adding an op to trigger all the tests internally and in CI)DOC");

REGISTER_CPU_OPERATOR(NotForCommit, NotForCommitOp);

} // namespace caffe2
