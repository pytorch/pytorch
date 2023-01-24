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
  template <class... Args>
  explicit IntegralImageOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;
};

template <typename T, class Context>
class IntegralImageGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit IntegralImageGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  Tensor row_pass_buffer_;
};

} // namespace caffe2

#endif // INTEGRAL_IMAGE_OP_H_
