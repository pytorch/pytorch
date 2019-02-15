#ifndef INTEGRAL_IMAGE_OP_H_
#define INTEGRAL_IMAGE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class IntegralImageOp final : public Operator<Context> {
 public:
  IntegralImageOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
};

template <typename T, class Context>
class IntegralImageGradientOp final : public Operator<Context> {
 public:
  IntegralImageGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  Tensor row_pass_buffer_;
};

} // namespace caffe2

#endif // INTEGRAL_IMAGE_OP_H_
