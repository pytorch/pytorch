#ifndef BATCHPERMUTATION_OP_H_
#define BATCHPERMUTATION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(BatchPermutation)

namespace caffe2 {

template <typename T, class Context>
class BatchPermutationOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit BatchPermutationOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice();
};

template <typename T, class Context>
class BatchPermutationGradientOp final : public Operator<Context> {
 public:
  BatchPermutationGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice();
};

} // namespace caffe2

#endif // BATCHPERMUTATION_OP_H_
