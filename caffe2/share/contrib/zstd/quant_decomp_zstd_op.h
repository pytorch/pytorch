#ifndef QUANT_DECOMP_OP_H_
#define QUANT_DECOMP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// Decompress a set of tensors compressed using zstd,
// see quant_decomp_op_test.py for how to compress
class QuantDecompZstdOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  QuantDecompZstdOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  ~QuantDecompZstdOp() {}

  bool RunOnDevice() override;
};

} // namespace caffe2
#endif // QUANT_DECOMP_OP_H_
