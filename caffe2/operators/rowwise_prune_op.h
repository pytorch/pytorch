#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

// indictor type I
// tensor value type V
template <typename I, typename V, typename Context>
class RowwisePruneOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RowwisePruneOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        abs_(this->template GetSingleArgument<bool>("abs", false)) {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(
        Input(THD).numel(),
        1,
        "Input `threshold` should have exactly 1 element, got: ",
        Input(THD).numel());
    CAFFE_ENFORCE_EQ(
        Input(X).dim(),
        2,
        "Input `X` should have exactly 2 dims, got: ",
        Input(X).dim());
    CAFFE_ENFORCE_EQ(
        Input(X).size(0),
        Input(INDICATOR).numel(),
        "Number of rows in `X` should be same to the numel of indicator, got: ",
        Input(X).size(0),
        " vs. ",
        Input(INDICATOR).numel());

    CAFFE_ENFORCE_EQ(
        Input(INDICATOR).dtype(),
        Input(THD).dtype(),
        "Indicator and threshold must have the same type, got: ",
        Input(INDICATOR).dtype().name(),
        " and",
        Input(THD).dtype().name());

    auto* x_out = Output(X_OUT)->template mutable_data<V>();
    CAFFE_ENFORCE_EQ(
        &Input(X), Output(X_OUT), "In place operation is required");
    CAFFE_ENFORCE_NE(x_out, (void*)nullptr, "output data cannot be null");

    const auto thd = Input(THD).template data<I>()[0];
    const auto* indicator = Input(INDICATOR).template data<I>();

    auto* compressed_idx =
        Output(CMP_IDX, Input(INDICATOR).numel(), at::dtype<int64_t>());
    auto* compressed_idx_data =
        compressed_idx->template mutable_data<int64_t>();

    const auto num_cols = Input(X).size(1);
    int64_t num_kept = 0;
    for (int64_t i = 0; i < Input(INDICATOR).numel(); i++) {
      const I val = abs_ ? std::abs(indicator[i]) : indicator[i];
      if (val <= thd) {
        compressed_idx_data[i] = -1;
        continue;
      }
      if (i != num_kept) {
        memcpy(
            x_out + num_kept * num_cols,
            x_out + i * num_cols,
            num_cols * sizeof(V));
      }
      compressed_idx_data[i] = num_kept;
      num_kept++;
    }
    Output(X_OUT)->Resize(num_kept, num_cols);
    return true;
  }

 protected:
  bool abs_;
  INPUT_TAGS(X, INDICATOR, THD);
  OUTPUT_TAGS(X_OUT, CMP_IDX);
};

} // namespace caffe2
