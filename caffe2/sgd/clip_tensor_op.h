#ifndef CAFFE2_OPERATORS_CLIP_TENSOR_OP_H_
#define CAFFE2_OPERATORS_CLIP_TENSOR_OP_H_

#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename Context>
class ClipTensorByScalingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ClipTensorByScalingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    threshold_ = this->template GetSingleArgument<float>("threshold", 0.0);
    CAFFE_ENFORCE_GT(threshold_, 0, "Threshold must be greater than 0");
  }

  bool RunOnDevice() override {
    const auto& input_tensor = Input(0);
    CAFFE_ENFORCE_GT(input_tensor.numel(), 0);
    const auto& val = Input(1);
    CAFFE_ENFORCE_EQ(val.numel(), 1);

    const auto* input_tensor_data = input_tensor.template data<float>();
    const auto* val_data = val.template data<float>();

    auto* clipped = Output(0, input_tensor.sizes(), at::dtype<float>());
    float* clipped_tensor_data = clipped->template mutable_data<float>();

    if (InputSize() > 2) {
      const auto& additional_threshold = Input(2);
      CAFFE_ENFORCE_EQ(additional_threshold.numel(), 1);

      threshold_ *= *(additional_threshold.template data<float>());
    }

    if (*val_data > threshold_) {
      float ratio = threshold_ / *val_data;

      math::Scale<float, float, Context>(
          clipped->numel(),
          ratio,
          input_tensor_data,
          clipped_tensor_data,
          &context_);
    } else {
      if (input_tensor_data != clipped_tensor_data) {
        clipped->CopyFrom(input_tensor, /*async*/ true);
      }
    }

    return true;
  }

 private:
  float threshold_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CLIP_TENSOR_OP_H_
