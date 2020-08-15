
#ifndef CAFFE2_OPERATORS_LENGTHS_TOP_K_OP_H_
#define CAFFE2_OPERATORS_LENGTHS_TOP_K_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <typename T, class Context>
class LengthsTopKOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit LengthsTopKOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
  }

  bool RunOnDevice() override;

 protected:
  int k_;
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(TOPK_VALUES_OUT, TOPK_INDICES_OUT);
};

template <typename T, class Context>
class LengthsTopKGradientOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit LengthsTopKGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int k_;
  INPUT_TAGS(LENGTH_IN, INDICES_IN, DER_TOPK_IN);
  OUTPUT_TAGS(DER_X_OUT);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_TOP_K_OP_H_
