#if defined(__linux__)

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include <csignal>

namespace caffe2 {

class CrashOp final : public Operator<CPUContext> {
 public:
  CrashOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override {
    raise(SIGABRT);
    return true;
  }
};

OPERATOR_SCHEMA(Crash).NumInputs(0).NumOutputs(0).SetDoc(
    R"DOC(Crashes the program. Use for testing)DOC");

REGISTER_CPU_OPERATOR(Crash, CrashOp);

} // namespace caffe2
#endif
