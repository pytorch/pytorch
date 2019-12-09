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
  template <class... Args>
  explicit PercentileOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}

  bool RunOnDevice() override;

 protected:
  INPUT_TAGS(X, VAL_PCT_PAIRS, LENS);
  OUTPUT_TAGS(PCT);
  Tensor values_tensor;
  Tensor percentiles_tensor;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PERCENTILE_OP_H_
