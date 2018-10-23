#ifndef CAFFE2_OPERATORS_LABEL_GAIN_MAP_OP_H_
#define CAFFE2_OPERATORS_LABEL_GAIN_MAP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class LabelGainMapOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  LabelGainMapOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        map_lengths_(OperatorBase::GetRepeatedArgument<int>(
            "map_lengths",
            vector<int>{})),
        map_keys_(OperatorBase::GetRepeatedArgument<float>(
            "map_keys",
            vector<float>{})),
        map_values_(OperatorBase::GetRepeatedArgument<float>(
            "map_values",
            vector<float>{})) {
    n_maps = map_lengths_.size();
    CAFFE_ENFORCE_GT(n_maps, 0, "Provide at least one map");
    CAFFE_ENFORCE_EQ(
        map_keys_.size(),
        map_values_.size(),
        "The sizes of the map keys and map values must match.");

    CAFFE_ENFORCE_EQ(
        std::accumulate(map_lengths_.begin(), map_lengths_.end(), 0),
        map_keys_.size(),
        "The length information and the map keys length must match.");

    int index = 0;
    for (int i_m = 0; i_m < n_maps; ++i_m) {
      vector<float> keys_i;
      vector<float> vals_i;
      for (int j = 0; j < map_lengths_[i_m]; ++j) {
        keys_i.push_back(map_keys_[index]);
        vals_i.push_back(map_values_[index]);
        ++index;
      }
      keys_.push_back(keys_i);
      vals_.push_back(vals_i);
    }
  }

  bool RunOnDevice() override {
    // Input
    const auto& keys = Input(KEYS);
    CAFFE_ENFORCE_EQ(keys.ndim(), 2);
    CAFFE_ENFORCE_EQ(
        keys.dim(1),
        n_maps,
        "The number of element in each row should match the number of maps");
    const auto batch_size = keys.dim(0);
    const float* keys_data = keys.template data<float>();

    // Output
    auto* values = Output(VALUES);
    values->ResizeLike(keys);
    float* values_output = values->template mutable_data<float>();

    int cur_index = -1;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < n_maps; ++j) {
        ++cur_index;
        float cur_key = keys_data[cur_index];
        // Check value range
        CAFFE_ENFORCE(
            cur_key >= keys_[j].front() && cur_key <= keys_[j].back(),
            "The key is out of range: ",
            cur_key,
            " vs ",
            "[",
            keys_[j].front(),
            ", ",
            keys_[j].back(),
            "]");
        const auto lower_bound =
            std::lower_bound(keys_[j].begin(), keys_[j].end(), cur_key);
        const int pos = std::distance(keys_[j].begin(), lower_bound);
        if (keys_[j][pos] == cur_key) {
          values_output[cur_index] = vals_[j][pos];
          continue;
        }
        // interpolation
        float w = (cur_key - keys_[j][pos - 1]) /
            (keys_[j][pos] - keys_[j][pos - 1] + kEPSILON);
        values_output[cur_index] =
            (1 - w) * vals_[j][pos - 1] + w * vals_[j][pos];
      }
    }
    return true;
  }

 protected:
  INPUT_TAGS(KEYS);
  OUTPUT_TAGS(VALUES);

 private:
  int n_maps;
  vector<int> map_lengths_;
  vector<float> map_keys_;
  vector<float> map_values_;
  vector<vector<float>> keys_;
  vector<vector<float>> vals_;

  const float kEPSILON = 1e-10;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LABEL_GAIN_MAP_OP_H_
