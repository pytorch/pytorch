#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class TORCH_API SparseNormalizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit SparseNormalizeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        use_max_norm_(
            this->template GetSingleArgument<bool>("use_max_norm", true)),
        norm_(this->template GetSingleArgument<float>("norm", 1.0)) {
    CAFFE_ENFORCE_GE(norm_, 0, "norm should be bigger than 0");
  }

  bool RunOnDevice() override;

  template <typename SIndex>
  bool DoRunWithType();

 protected:
  bool use_max_norm_;
  float norm_;
  INPUT_TAGS(PARAM, INDICES);
  OUTPUT_TAGS(OUTPUT_PARAM);
};

} // namespace caffe2
