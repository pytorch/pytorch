#ifndef CAFFE2_OPERATORS_BISECT_PERCENTILE_OP_H_
#define CAFFE2_OPERATORS_BISECT_PERCENTILE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class BisectPercentileOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  BisectPercentileOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        pct_raw_(OperatorBase::GetRepeatedArgument<float>(
            "percentile_raw",
            vector<float>{})),
        pct_mapping_(OperatorBase::GetRepeatedArgument<float>(
            "percentile_mapping",
            vector<float>{})),
        pct_lower_(OperatorBase::GetRepeatedArgument<float>(
            "percentile_lower",
            vector<float>{})),
        pct_upper_(OperatorBase::GetRepeatedArgument<float>(
            "percentile_upper",
            vector<float>{})),
        pct_lens_(
            OperatorBase::GetRepeatedArgument<int>("lengths", vector<int>{})) {
    CAFFE_ENFORCE_EQ(
        pct_raw_.size(),
        pct_mapping_.size(),
        "Feature (raw) data and percentile value dimension should match.");
    CAFFE_ENFORCE_EQ(
        pct_raw_.size(),
        pct_lower_.size(),
        "Feature (raw) data and lower bound dimension should match.");
    CAFFE_ENFORCE_EQ(
        pct_raw_.size(),
        pct_upper_.size(),
        "Feature (raw) data and upper bound dimension should match.");
    n_features = pct_lens_.size();
    index.reserve(n_features + 1);
    index[0] = 0;
    for (int i = 1; i <= n_features; ++i) {
      index[i] = index[i - 1] + pct_lens_[i - 1];
    }
    CAFFE_ENFORCE_EQ(
        index[n_features], // The sum of lengths_data
        pct_raw_.size(),
        "Sum of lengths should be equal to the total number of percentile "
        "mapping data samples");
  }

  bool RunOnDevice() override {
    // Input
    const auto& raw = Input(RAW);
    CAFFE_ENFORCE_EQ(raw.dim(), 2);
    const auto batch_size = raw.size(0);
    const auto num_features = raw.size(1);
    CAFFE_ENFORCE_EQ(num_features, pct_lens_.size());
    const float* raw_data = raw.template data<float>();

    // Output
    auto* pct = Output(PCT);
    pct->ResizeLike(raw);
    float* pct_output = pct->template mutable_data<float>();

    // Compute percentile for each raw feature value
    int feature_start_index = 0;
    int feature_length = 0;
    int cur_index = 0;

    for (int i = 0; i < num_features; ++i) {
      cur_index = i;
      feature_start_index = index[i];
      feature_length = pct_lens_[i];
      for (int j = 0; j < batch_size; ++j) {
        pct_output[cur_index] = compute_percentile(
            pct_raw_.begin() + feature_start_index,
            pct_mapping_.begin() + feature_start_index,
            pct_lower_.begin() + feature_start_index,
            pct_upper_.begin() + feature_start_index,
            feature_length,
            raw_data[cur_index]);
        cur_index += num_features;
      }
    }
    return true;
  }

 protected:
  INPUT_TAGS(RAW);
  OUTPUT_TAGS(PCT);

 private:
  int n_features;
  vector<float> pct_raw_;
  vector<float> pct_mapping_;
  vector<float> pct_lower_;
  vector<float> pct_upper_;
  vector<int> pct_lens_;
  vector<int> index;
  vector<std::map<float, float>> fast_pct;

  const float kEPSILON = 1e-10;

  int binary_search(
      const std::vector<float>::iterator& data,
      int lo,
      int hi,
      float val) {
    int mid;
    bool low_cond, high_cond;

    while (lo < hi) {
      mid = (lo + hi) >> 1;
      low_cond = (data[mid] <= val);
      high_cond = (val < data[mid + 1]);
      if (low_cond && high_cond) {
        return mid;
      } else if (!low_cond) {
        hi = mid - 1;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  float compute_percentile(
      const std::vector<float>::iterator& pct_raw_it,
      const std::vector<float>::iterator& pct_mapping_it,
      const std::vector<float>::iterator& pct_lower_it,
      const std::vector<float>::iterator& pct_upper_it,
      const int size,
      const float val) {
    // Corner cases where no interpolation is needed.
    if (val < pct_raw_it[0]) {
      return 0.;
    }
    if (val > pct_raw_it[size - 1]) {
      return 1.;
    }

    float result;
    // Interpolation by binary search
    const auto k = binary_search(pct_raw_it, 0, size - 1, val);

    if (pct_raw_it[k] == val) {
      // Exact match
      result = pct_mapping_it[k];
    } else {
      // interpolation
      float w = (val - pct_raw_it[k]) /
          (pct_raw_it[k + 1] - pct_raw_it[k] + kEPSILON);
      result = (1 - w) * pct_upper_it[k] + w * pct_lower_it[k + 1];
    }
    return result;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BISECT_PERCENTILE_OP_H_
