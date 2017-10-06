// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef CAFFE2_OPERATORS_WEIGHTEDSAMPLE_OP_H_
#define CAFFE2_OPERATORS_WEIGHTEDSAMPLE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class WeightedSampleOp final : public Operator<Context> {
 public:
  WeightedSampleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& weights = Input(0);
    int batch_size = weights.dim(0);
    int weights_dim = weights.dim(1);
    auto* output = Output(0);

    if (batch_size > 0 && weights_dim > 0) {
      output->Resize(batch_size, 1);
      cum_mass_.resize(weights_dim);
      const T* mat_weights = weights.template data<T>();
      T* output_indices = output->template mutable_data<T>();

      for (int i = 0; i < batch_size; i++) {
        offset_ = i * weights_dim;

        for (int j = 0; j < weights_dim; j++) {
          if (j == 0) {
            cum_mass_[j] = mat_weights[offset_ + j];
          } else {
            cum_mass_[j] = cum_mass_[j - 1] + mat_weights[offset_ + j];
          }
        }

        math::RandUniform<float, Context>(
            1, 0.0f, cum_mass_[cum_mass_.size() - 1], &r_, &context_);
        // Makes the element in cum_mass_ slightly bigger
        // to compensate inaccuracy introduced due to rounding,
        cum_mass_[cum_mass_.size() - 1] += 0.01f;
        auto lb = lower_bound(cum_mass_.begin(), cum_mass_.end(), r_);
        CAFFE_ENFORCE(
            lb != cum_mass_.end(), "Cannot find ", r_, " in cum_mass_.");
        output_indices[i] = static_cast<int>(lb - cum_mass_.begin());
      }
    } else {
      output->Resize(0);
      output->template mutable_data<T>();
    }

    return true;
  }

 private:
  vector<float> cum_mass_;
  float r_;
  int offset_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_WEIGHTEDSAMPLE_OP_H_
