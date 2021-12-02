#pragma once

#include "caffe2/core/operator.h"

namespace caffe2 {

class RowWiseCounterOp final : public Operator<CPUContext> {
 public:
  RowWiseCounterOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws),
        counter_halflife_(
            this->template GetSingleArgument<int64_t>("counter_halflife", -1)),
        counter_neg_log_rho_(0.0) {
    if (counter_halflife_ > 0) {
      counter_neg_log_rho_ = std::log(2.0) / counter_halflife_;
    }
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(Input(PREV_ITER).numel(), Input(COUNTER).numel());
    CAFFE_ENFORCE_EQ(Input(ITER).numel(), 1);
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType() {
    auto* prev_iter =
        Output(OUTPUT_PREV_ITER)->template mutable_data<int64_t>();
    auto* counter = Output(OUTPUT_COUNTER)->template mutable_data<double>();

    const int64_t curr_iter = Input(ITER).template data<int64_t>()[0];
    const auto* indices = Input(INDICES).template data<SIndex>();

    auto n = Input(INDICES).numel();
    if (n == 0) {
      return true;
    }
    if (counter_halflife_ <= 0) {
      return true;
    }

    for (auto i = 0; i < n; ++i) {
      const std::size_t idx = indices[i];
      CAFFE_ENFORCE_GE(
          Input(COUNTER).numel(),
          idx,
          this->debug_def().input(COUNTER),
          ", out of bound,  idx:",
          idx,
          " for input i:",
          i,
          " max size:",
          Input(COUNTER).numel());
      const int64_t iter_delta =
          std::max<int64_t>(0, curr_iter - prev_iter[idx]);

      counter[idx] =
          1.0 + std::exp(-iter_delta * counter_neg_log_rho_) * counter[idx];
      prev_iter[idx] = std::max<int64_t>(curr_iter, prev_iter[idx]);
    }
    return true;
  }

 protected:
  int64_t counter_halflife_;
  double counter_neg_log_rho_;
  INPUT_TAGS(PREV_ITER, COUNTER, INDICES, ITER);
  OUTPUT_TAGS(OUTPUT_PREV_ITER, OUTPUT_COUNTER);
};

} // namespace caffe2
