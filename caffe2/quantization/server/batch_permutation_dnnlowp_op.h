#pragma once

#include "caffe2/operators/copy_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

// FIXME
using BatchPermutationFP32Op = CopyOp<CPUContext, CPUContext, CPUContext>;

template <typename T>
class BatchPermutationDNNLowPOp final
    : public DNNLowPOp<T, BatchPermutationFP32Op> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, BatchPermutationFP32Op);

  BatchPermutationDNNLowPOp(const OperatorDef& operator_def, Workspace* ws)
      : BaseType(operator_def, ws) {}

  bool RunOnDevice() override;

 private:
  INPUT_TAGS(INPUT, INDICES);
  OUTPUT_TAGS(OUTPUT);
};

} // namespace caffe2
