#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "c10/util/irange.h"

#include <vector>

namespace caffe2 {
template <typename F, typename T, class Context>
class NGramFromCategoricalOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit NGramFromCategoricalOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        col_ids_(this->template GetRepeatedArgument<int>("col_ids")),
        categorical_limits_(
            this->template GetRepeatedArgument<int>("categorical_limits")),
        vals_(this->template GetRepeatedArgument<int>("vals")) {
    col_num_ = col_ids_.size();
    max_col_id_ = *std::max_element(col_ids_.begin(), col_ids_.end());
    CAFFE_ENFORCE_EQ(col_num_, categorical_limits_.size());
    int expected_vals_size = 0;
    for (auto& l : categorical_limits_) {
      CAFFE_ENFORCE_GT(l, 0);
      expected_vals_size += l;
    }
    CAFFE_ENFORCE_EQ(expected_vals_size, vals_.size());
    // compute ngram maps with small end
    for (auto& j : col_ids_) {
      CAFFE_ENFORCE_GE(j, 0);
      ngram_maps_.push_back(std::map<int, int>());
    }
    int base = 1;
    int idx = 0;
    for (const auto k : c10::irange(col_num_)) {
      int l = categorical_limits_[k];
      for (const auto m : c10::irange(l)) {
        int v = vals_[idx++];
        ngram_maps_[k][v] = m * base;
      }
      base *= l;
    }
  }

  bool RunOnDevice() override {
    auto& floats = Input(0);
    auto N = floats.size(0);
    auto D = floats.size_from_dim(1);
    const F* floats_data = floats.template data<F>();

    auto* output = Output(0, {N}, at::dtype<T>());
    auto* output_data = output->template mutable_data<T>();
    math::Set<T, Context>(output->numel(), 0, output_data, &context_);

    CAFFE_ENFORCE_GT(D, max_col_id_);
    for (const auto i : c10::irange(N)) {
      for (const auto k : c10::irange(col_num_)) {
        int j = col_ids_[k];
        int v = round(floats_data[i * D + j]);
        // for out-of-vocabulary values, we always treat them the same as the
        // first value specified in vals; if we want to mimic the behavior as
        // sigrid NGram transform, just push front a random/impossible value at
        // each segments of vals
        output_data[i] += ngram_maps_[k].find(v) == ngram_maps_[k].end()
            ? 0
            : ngram_maps_[k][v];
      }
    }
    return true;
  }

 private:
  std::vector<int> col_ids_;
  std::vector<int> categorical_limits_;
  std::vector<int> vals_;
  std::vector<std::map<int, int>> ngram_maps_;
  int col_num_;
  int max_col_id_;
};
} // namespace caffe2
