#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// IterOp runs an iteration counter. I cannot think of a case where we would
// need to access the iter variable on device, so this will always produce an
// int value as its output.
class IterOp final : public OperatorBase {
 public:
  IterOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws), iter_(-1) {}

  bool Run() override {
    iter_++;
    *OperatorBase::Output<int>(0) = iter_;
    return true;
  }

 private:
  int iter_;
  INPUT_OUTPUT_STATS(0, 0, 1, 1);
  DISABLE_COPY_AND_ASSIGN(IterOp);
};

namespace {
REGISTER_CPU_OPERATOR(Iter, IterOp)
REGISTER_CUDA_OPERATOR(Iter, IterOp)
}
}  // namespace caffe2
