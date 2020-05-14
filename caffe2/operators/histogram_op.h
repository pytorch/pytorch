#pragma once

#include <cmath>
#include <limits>
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class HistogramOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit HistogramOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        bin_edges_(this->template GetRepeatedArgument<float>("bin_edges")) {
    CAFFE_ENFORCE_GE(
        bin_edges_.size(),
        2,
        "Number of bin edges must be greater than or equal to 2.");
    for (int i = 1; i < bin_edges_.size(); i++) {
      CAFFE_ENFORCE_GT(
          bin_edges_[i],
          bin_edges_[i - 1],
          "bin_edges must be a strictly increasing sequence of values.");
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    CheckInputs();

    const auto* histogram = Output(HISTOGRAM);
    histogram->Resize(bin_edges_.size() - 1);
    auto* histogram_data = histogram->template mutable_data<int64_t>();
    math::Set<int64_t, Context>(
        bin_edges_.size() - 1, 0, histogram_data, &context_);

    for (int input_idx = 0; input_idx < InputSize(); input_idx++) {
      const auto& x = Input(input_idx);
      const int64_t N = x.numel();
      const auto* x_data = x.template data<T>();
      for (int data_idx = 0; data_idx < N; data_idx++) {
        const int bisection_idx = BisectBinEdges(x_data[data_idx]);
        if (bisection_idx > 0 && bisection_idx < bin_edges_.size()) {
          histogram_data[bisection_idx - 1]++;
        }
      }
    }

    return true;
  }

 protected:
  OUTPUT_TAGS(HISTOGRAM);

 private:
  vector<float> bin_edges_;

  void CheckInputs() {
    const auto& input_zero = Input(0);
    for (int i = 1; i < InputSize(); i++) {
      CAFFE_ENFORCE_EQ(
          Input(i).dtype(),
          input_zero.dtype(),
          "All inputs must have the same type; expected ",
          input_zero.dtype().name(),
          " but got ",
          Input(i).dtype().name(),
          " for input ",
          i);
    }
  }

  template <typename T>
  int BisectBinEdges(const T& value) {
    int hi = bin_edges_.size();
    int lo = 0;
    while (lo < hi) {
      int mid = (hi + lo) / 2;
      if (value < bin_edges_[mid]) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }
};

} // namespace caffe2
