#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class TORCH_API SparseLpRegularizerOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SparseLpRegularizerOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        p_(this->template GetSingleArgument<float>("p", 2.0)),
        reg_lambda_(
            this->template GetSingleArgument<float>("reg_lambda", 1e-5)) {
    CAFFE_ENFORCE(
        p_ == 1.0 || p_ == 2.0,
        "Sparse Lp regularizer only implemented for p=1 or p=2.");
    CAFFE_ENFORCE_GT(
        reg_lambda_,
        0.0,
        "Lambda for sparse Lp regularizer must be greater than 0.");
    CAFFE_ENFORCE_LT(
        reg_lambda_,
        1.0,
        "Lambda for sparse Lp regularizer must be less than 1.");
  }

  bool RunOnDevice() override;

  template <typename SIndex>
  bool DoRunWithType();

 protected:
  float p_;
  float reg_lambda_;
  INPUT_TAGS(PARAM, INDICES);
  OUTPUT_TAGS(OUTPUT_PARAM);
};

} // namespace caffe2
