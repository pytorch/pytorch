#ifndef CAFFE2_OPERATORS_SPARSE_ITEMWISE_DROPOUT_WITH_REPLACEMENT_OP_H_
#define CAFFE2_OPERATORS_SPARSE_ITEMWISE_DROPOUT_WITH_REPLACEMENT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class SparseItemwiseDropoutWithReplacementOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SparseItemwiseDropoutWithReplacementOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        ratio_(this->template GetSingleArgument<float>("ratio", 0.0)),
        replacement_value_(
            this->template GetSingleArgument<int64_t>("replacement_value", 0)) {
    // It is allowed to drop all or drop none.
    CAFFE_ENFORCE_GE(ratio_, 0.0, "Ratio should be a valid probability");
    CAFFE_ENFORCE_LE(ratio_, 1.0, "Ratio should be a valid probability");
  }

  bool RunOnDevice() override;

 private:
  float ratio_;
  int64_t replacement_value_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPARSE_ITEMWISE_DROPOUT_WITH_REPLACEMENT_OP_H_
