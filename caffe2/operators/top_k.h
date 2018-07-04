#ifndef CAFFE2_OPERATORS_TOP_K_H_
#define CAFFE2_OPERATORS_TOP_K_H_

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class TopKOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1),
        OP_SINGLE_ARG(int, "axis", axis_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKOp() {}

  bool RunOnDevice() override;

 private:
  const int k_;
  int axis_;
};

template <typename T, class Context>
class TopKGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, -1) {}

  ~TopKGradientOp() {}

  bool RunOnDevice() override;

 private:
  int axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TOP_K_H_
