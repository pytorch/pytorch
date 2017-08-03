#ifndef CAFFE2_OPERATORS_KMAX_POOLING_OP_H_
#define CAFFE2_OPERATORS_KMAX_POOLING_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <typename T, class Context>
class kMaxPoolingOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  kMaxPoolingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(InputSize(), 2, "input size must be 2");
    auto& X = Input(X_IN);
    auto& Y = Input(Y_IN);
    int N = Y.dim(0);
    const T* X_data = X.template data<T>();
    const int* input_len = Y.template data<int>();
    auto* output_kmax_values = Output(kMAX_VALUES_OUT);
    auto* output_kmax_indices = Output(kMAX_INDICES_OUT);

    output_kmax_values->Resize(N * k_);
    output_kmax_indices->Resize(N * k_);
    std::vector<int> output_dims = std::vector<int>({N, k_});
    output_kmax_values->Reshape(output_dims);
    output_kmax_indices->Reshape(output_dims);
    T* output_kmax_values_data = output_kmax_values->template mutable_data<T>();
    int* output_kmax_indices_data =
        output_kmax_indices->template mutable_data<int>();

    auto cmp = [](std::pair<T, TIndex>& lhs, std::pair<T, TIndex>& rhs) {
      return lhs.first > rhs.first ||
          (lhs.first == rhs.first && lhs.second < rhs.second);
    };

    // Sort preserving indices
    int next_index = 0;
    for (TIndex i = 0; i < N; ++i) {
      // Build a min-heap, the heap element is pair of (value, idx)
      // the top of the heap is the smallest value
      std::priority_queue<
          std::pair<T, TIndex>,
          std::vector<std::pair<T, TIndex>>,
          decltype(cmp)>
          PQ(cmp);

      // Maintain the size of heap to be less or equal to k_, so the
      // heap will hold the k_ largest values
      for (TIndex j = 0; j < input_len[i]; ++j) {
        const auto value = X_data[next_index++];
        if (PQ.size() < k_ || value > PQ.top().first) {
          PQ.push(std::make_pair(value, j));
        }
        if (PQ.size() > k_) {
          PQ.pop();
        }
      }

      int last_index = PQ.size();
      for (TIndex j = 0; j < k_; ++j) {
        if (PQ.size() > 0) {
          auto& pqElem = PQ.top();
          output_kmax_values_data[i * k_ + last_index - j - 1] = pqElem.first;
          output_kmax_indices_data[i * k_ + last_index - j - 1] = pqElem.second;
          PQ.pop();
        } else {
          output_kmax_values_data[i * k_ + j] = 0;
          output_kmax_indices_data[i * k_ + j] = -1;
        }
      }
    }

    return true;
  }

 protected:
  int k_;
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(kMAX_VALUES_OUT, kMAX_INDICES_OUT);
};

template <typename T, class Context>
class kMaxPoolingGradientOp : public Operator<Context> {
 public:
  kMaxPoolingGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(InputSize(), 3, "input size must be 3");
    auto& input_len = Input(LENGTH_IN);
    int N = input_len.size();
    auto& input_indices = Input(INDICES_IN);
    CAFFE_ENFORCE_GE(input_indices.ndim(), 2, "input dim must be >= 2");
    CAFFE_ENFORCE_EQ(
        input_indices.size(), N * k_, "input_indices shape is not correct");
    auto& input_kmax = Input(DER_kMAX_IN);
    CAFFE_ENFORCE_EQ(
        input_kmax.size(), N * k_, "input_kmax shape is not correct");
    auto* X_out = Output(DER_X_OUT);

    const int* input_len_data = input_len.template data<int>();
    const int* input_indices_data = input_indices.template data<int>();
    const T* input_kmax_data = input_kmax.template data<T>();

    int num_indices = 0;
    for (int i = 0; i < N; i++) {
      num_indices += input_len_data[i];
    }
    X_out->Resize(num_indices);
    std::vector<int> output_dims = std::vector<int>({num_indices});
    X_out->Reshape(output_dims);
    T* X_out_data = X_out->template mutable_data<T>();
    math::Set<T, Context>(num_indices, 0.0, X_out_data, &context_);

    int index_offset = 0;
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < std::min(input_len_data[i], k_); j++) {
        int cur_index = index_offset + input_indices_data[i * k_ + j];
        CAFFE_ENFORCE_LE(
            cur_index, num_indices, "cur_index should <= num_indices");
        X_out_data[cur_index] = input_kmax_data[i * k_ + j];
      }
      index_offset += input_len_data[i];
    }

    return true;
  }

 protected:
  int k_;
  INPUT_TAGS(LENGTH_IN, INDICES_IN, DER_kMAX_IN);
  OUTPUT_TAGS(DER_X_OUT);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_KMAX_POOLING_OP_H_
