#ifndef CAFFE2_OPERATORS_LABEL_OPS_H_
#define CAFFE2_OPERATORS_LABEL_OPS_H_

#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SparseLabelSplitOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  SparseLabelSplitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_labels_(this->template GetSingleArgument<int>("num_labels", -1)) {
    // num_labels is optional: if it provided we run-time check if it's
    // consistent with the output size; otherwise we infer from the output
    if (num_labels_ != -1) {
      CAFFE_ENFORCE(
          operator_def.output_size() == 2 * num_labels_ ||
          operator_def.output_size() == 2 * num_labels_ + 1);
    } else {
      // odd number of outputs: offset map is used
      // even number of outputs: offset map is not used
      num_labels_ = operator_def.output_size() / 2;
    }
  }

  bool RunOnDevice() override {
    auto& len = Input(0);
    auto& label_ind = Input(1);
    auto& label_val = Input(2);
    const auto* len_data = len.template data<int32_t>();
    const auto* label_ind_data = label_ind.template data<int64_t>();
    const T* label_val_data = label_val.template data<T>();

    auto N_len = len.dim(0);
    auto N = label_ind.dim(0);

    CAFFE_ENFORCE_EQ(
        label_val.dim(0),
        N,
        "label_index should have the same length as label_value");

    CAFFE_ENFORCE_EQ(
        std::accumulate(len_data, len_data + N_len, 0),
        N,
        "The sum of length should be equal to the length of other inputs");

    vector<int> n_example_per_task_(num_labels_, 0);
    for (int i = 0; i < N; i++) {
      auto label_id = label_ind_data[i];
      // label_id should start from 0
      CAFFE_ENFORCE_LT(label_id, num_labels_, "label_index out of range");
      CAFFE_ENFORCE_GE(label_id, 0, "label_index out of range");
      n_example_per_task_[label_id]++;
    }

    vector<T*> label_vec(num_labels_);
    vector<int*> eid_vec(num_labels_);
    for (int i = 0; i < num_labels_; i++) {
      auto* labels = Output(i, {n_example_per_task_[i]}, at::dtype<T>());
      auto* eids =
          Output(i + num_labels_, {n_example_per_task_[i]}, at::dtype<int>());
      label_vec[i] = labels->template mutable_data<T>();
      eid_vec[i] = eids->template mutable_data<int>();
    }

    int* offset_map_data;
    if (OutputSize() > 2 * num_labels_) {
      auto* offset_map = Output(2 * num_labels_, {N}, at::dtype<int>());
      offset_map_data = offset_map->template mutable_data<int>();
    } else {
      offset_map_data = nullptr;
    }

    std::fill(n_example_per_task_.begin(), n_example_per_task_.end(), 0);
    int pos = 0;
    for (int i = 0; i < N_len; i++) {
      auto cur_len = len_data[i];
      for (int l = 0; l < cur_len; l++) {
        auto ind = label_ind_data[pos];
        auto val = label_val_data[pos];
        auto pos_output = n_example_per_task_[ind]++;

        if (offset_map_data) {
          offset_map_data[pos] = pos_output;
        }
        label_vec[ind][pos_output] = val;
        eid_vec[ind][pos_output] = i;
        pos++;
      }
    }

    return true;
  }

 private:
  int num_labels_;
  // the following tensors are used in the GPU case
  Tensor eid_map_buffer_{Context::GetDeviceType()};
  Tensor label_output_ptr_buffer_{Context::GetDeviceType()};
  Tensor eid_output_ptr_buffer_{Context::GetDeviceType()};
};

template <typename T, class Context>
class SparseLabelSplitGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  SparseLabelSplitGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        num_labels_(this->template GetSingleArgument<int>("num_labels", 0)) {
    CAFFE_ENFORCE_GT(num_labels_, 0, "num_labels should be positive");
  }

  bool RunOnDevice() override {
    auto& len = Input(0);
    auto& label_ind = Input(1);
    const auto* len_data = len.template data<int32_t>();
    const auto* label_ind_data = label_ind.template data<int64_t>();

    const auto* offset_map_data = InputSize() > num_labels_ + 2
        ? Input(num_labels_ + 2).template data<int32_t>()
        : nullptr;

    auto N_len = len.dim(0);
    auto N = label_ind.dim(0);

    CAFFE_ENFORCE_EQ(
        std::accumulate(len_data, len_data + N_len, 0),
        N,
        "The sum of length should be equal to the length of other inputs");

    vector<const T*> val_grad_vec(num_labels_);
    vector<int> n_example_per_task_(num_labels_, 0);
    for (int i = 0; i < num_labels_; i++) {
      auto& val_grad = Input(i + 2);
      val_grad_vec[i] = val_grad.template data<T>();
      n_example_per_task_[i] = val_grad.numel();
    }

    auto* output = Output(0, label_ind.sizes(), at::dtype<T>());
    auto* output_data = output->template mutable_data<T>();

    vector<int> task_val_grad_offset_(num_labels_, 0);

    for (int pos = 0; pos < N; pos++) {
      auto ind = label_ind_data[pos];

      CAFFE_ENFORCE_LE(
          ind,
          num_labels_,
          "Label index is too large or number of inputs is too few");

      auto offset =
          offset_map_data ? offset_map_data[pos] : task_val_grad_offset_[ind]++;

      CAFFE_ENFORCE_LE(
          offset,
          n_example_per_task_[ind],
          "Not enough entries for value gradient");
      output_data[pos] = val_grad_vec[ind][offset];
    }

    return true;
  }

 private:
  int num_labels_;
  Tensor val_grad_ptr_buffer_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LABEL_OPS_H_
