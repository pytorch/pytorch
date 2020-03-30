#pragma once

#include <cmath>
#include <limits>
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename Context>
class QuantileOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  QuantileOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        quantile_(this->template GetSingleArgument<float>("quantile", -1.0)),
        abs_(this->template GetSingleArgument<bool>("abs", true)),
        tol_(this->template GetSingleArgument<float>("tol", 1e-3)) {
    CAFFE_ENFORCE_GE(
        quantile_,
        0,
        "input quantile should be ",
        "no less than 0, got ",
        quantile_);
    CAFFE_ENFORCE_GE(
        1.0f,
        quantile_,
        "input quantile should be ",
        "no larger than 1, got ",
        quantile_);
    CAFFE_ENFORCE_GT(
        tol_, 0, "tolerance should be ", "no less than 0, got ", tol_);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    Output(QUANTILE_VAL)->Resize(1);
    auto* quantile_val = Output(QUANTILE_VAL)->template mutable_data<T>();

    auto& input_zero = Input(0);
    int64_t numel = input_zero.numel();
    for (int i = 1; i < InputSize(); ++i) {
      CAFFE_ENFORCE_EQ(
          Input(i).dtype(),
          input_zero.dtype(),
          "All inputs must have the same type, expected: ",
          input_zero.dtype().name(),
          " but got: ",
          Input(i).dtype().name(),
          " for input: ",
          i);
      numel += Input(i).numel();
    }
    CAFFE_ENFORCE_GT(
        numel,
        0,
        "number of total element in input tensor should be ",
        "larger than 0, got ",
        numel);

    // the expected number of elements lessEq to the target value
    const int64_t target_cnt =
        static_cast<int64_t>(std::ceil(numel * quantile_));

    T hi = 0.0;
    T lo = 0.0;
    GetRangeFromInputs(&lo, &hi);
    if (target_cnt == 0) {
      // lowest possible value
      quantile_val[0] = lo;
      return true;
    }
    if (target_cnt == numel) {
      // highest possible value
      quantile_val[0] = hi;
      return true;
    }
    int64_t lo_cnt = CountLowerEq(lo);
    if (lo_cnt >= target_cnt) {
      // the target is one of the lowest value
      quantile_val[0] = lo;
      return true;
    }
    while (std::abs(hi - lo) > tol_ * (std::abs(hi) + std::abs(lo))) {
      // keep hi_cnt > target_idx and lo_cnt <= target_idx
      const T mid = lo + (hi - lo) / 2.0;
      const int64_t mid_cnt = CountLowerEq(mid);
      if (mid_cnt > target_cnt) {
        CAFFE_ENFORCE_NE(
            hi, mid, "numeric precision at limit, unable to continue bisect");
        hi = mid;
      } else if (mid_cnt < target_cnt) {
        CAFFE_ENFORCE_NE(
            lo, mid, "numeric precision at limit, unable to continue bisect");
        lo = mid;
      } else {
        // mid_cnt == target_cnt
        quantile_val[0] = mid;
        return true;
      }
    }
    quantile_val[0] = hi;
    return true;
  }

 protected:
  float quantile_;
  bool abs_;
  float tol_;
  OUTPUT_TAGS(QUANTILE_VAL);

  template <typename T>
  void GetRangeFromInputs(T* lo, T* hi) {
    *hi = std::numeric_limits<T>::lowest();
    *lo = std::numeric_limits<T>::max();
    for (int i = 0; i < InputSize(); ++i) {
      const auto* input = Input(i).template data<T>();
      for (int j = 0; j < Input(i).numel(); j++) {
        const T val = abs_ ? std::abs(input[j]) : input[j];
        if (*hi < val) {
          *hi = val;
        }
        if (*lo > val) {
          *lo = val;
        }
      }
    }
  }

  template <typename T>
  int64_t CountLowerEq(const T& thd) {
    int64_t count = 0;
    for (int i = 0; i < InputSize(); ++i) {
      const auto* input = Input(i).template data<T>();
      for (int j = 0; j < Input(i).numel(); j++) {
        const T val = abs_ ? std::abs(input[j]) : input[j];
        if (val <= thd) {
          count++;
        }
      }
    }
    return count;
  }
};

} // namespace caffe2
