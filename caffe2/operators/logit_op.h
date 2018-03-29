#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LogitGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LogitGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        eps_(OperatorBase::GetSingleArgument<float>("eps", 1e-6f)) {}
  ~LogitGradientOp() {}

  bool RunOnDevice() override;

 protected:
  float eps_;
};

} // namespace caffe2
