#ifndef CAFFE2_OPERATORS_LPNORM_OP_H_
#define CAFFE2_OPERATORS_LPNORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LpNormOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit LpNormOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "p", p_, 2),
        OP_SINGLE_ARG(bool, "average", average_, false) {
    CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
  }

  bool RunOnDevice() override;

 protected:
  const int p_;
  const bool average_;
};

template <typename T, class Context>
class LpNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit LpNormGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "p", p_, 2),
        OP_SINGLE_ARG(bool, "average", average_, false) {
    CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
  }

  bool RunOnDevice() override;

 protected:
  const int p_;
  const bool average_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LPNORM_OP_H_
