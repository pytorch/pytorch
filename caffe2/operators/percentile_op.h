// Operator to calculate percentile values for an input tensor of data,
// given samples of data from the same distribution, labeled with their
// percentile values.

#ifndef CAFFE2_OPERATORS_PERCENTILE_OP_H_
#define CAFFE2_OPERATORS_PERCENTILE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class PercentileOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  PercentileOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X, VAL_PCT_PAIRS, LENS);
  OUTPUT_TAGS(PCT);
  Tensor<Context> values_tensor;
  Tensor<Context> percentiles_tensor;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PERCENTILE_OP_H_
