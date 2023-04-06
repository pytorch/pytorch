#ifndef CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
#define CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(BatchBoxCox);

namespace caffe2 {

template <class Context>
class BatchBoxCoxOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit BatchBoxCoxOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        min_block_size_(
            this->template GetSingleArgument<int>("min_block_size", 256)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  std::size_t min_block_size_;

  INPUT_TAGS(DATA, LAMBDA1, LAMBDA2);
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_BATCH_BOX_COX_OPS_H_
