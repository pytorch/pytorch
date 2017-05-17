#ifndef CAFFE2_OPERATORS_NORMALIZE_OP_H_
#define CAFFE2_OPERATORS_NORMALIZE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class NormalizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  NormalizeOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() override;
};

template <typename T, class Context>
class NormalizeGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  NormalizeGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}

  bool RunOnDevice() override;

 private:
  INPUT_TAGS(INPUT, GRAD_OUT);
  OUTPUT_TAGS(GRAD_IN);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NORMALIZE_OP_H_
