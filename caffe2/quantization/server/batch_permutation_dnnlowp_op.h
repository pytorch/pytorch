#ifndef DEEPLEARNING_QUANTIZATION_CAFFE2_BATCH_PERMUTATION_DNNLOWP_OP_H_
#define DEEPLEARNING_QUANTIZATION_CAFFE2_BATCH_PERMUTATION_DNNLOWP_OP_H_

#include "caffe2/fb/operators/batch_permutation_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

using BatchPermutationFP32Op = BatchPermutationOp<float, CPUContext>;

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

#endif // DEEPLEARNING_QUANTIZATION_CAFFE2_BATCH_PERMUTATION_DNNLOWP_OP_H_
