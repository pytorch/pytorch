#ifndef CAFFE2_OPERATORS_TOP_K_H_
#define CAFFE2_OPERATORS_TOP_K_H_

#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
struct ValueCmp {
  bool operator()(
      const std::pair<T, TIndex>& lhs,
      const std::pair<T, TIndex>& rhs) {
    return (
        lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second));
  }
};

// Define these two names to allow lookup into the 2d tensors like
// mytensor(i, j)
template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

} // namespace

template <typename T, class Context>
class TopKOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* values = Output(0);
    auto* indices = Output(1);

    vector<TIndex> in_dims = input.dims();
    // Linearize input tensor except for last dimension
    // e.g. [3, 4, 5] -> [12, 5]
    // [5] -> [5]
    CAFFE_ENFORCE(
        in_dims.back() >= k_, "k argment should not be greater than last dim");
    vector<TIndex> linear_shape = {size_to_dim_(in_dims.size() - 1, in_dims),
                                   in_dims[in_dims.size() - 1]};
    auto input_map = ConstEigenMatrixMapRowMajor<T>(
        static_cast<const T*>(input.raw_data()),
        linear_shape[0],
        linear_shape[1]);

    // Resize output tensors to be the same shape as the linearized input except
    // for the last dimension, which will be of size k. E.x. for an input tensor
    // of shape [3, 4, 5] and k=2, both of these will be shape [3, 4, 2]
    vector<TIndex> output_linear_shape = {linear_shape[0], k_};
    values->Resize(output_linear_shape);
    indices->Resize(output_linear_shape);

    // Use Eigen maps to allow indexing into the 2d tensors like values_map(i,j)
    auto values_map = EigenMatrixMapRowMajor<T>(
        values->template mutable_data<T>(), linear_shape[0], k_);
    auto indices_map = EigenMatrixMapRowMajor<TIndex>(
        indices->template mutable_data<TIndex>(), linear_shape[0], k_);

    // Sort preserving indices
    for (TIndex i = 0; i < linear_shape[0]; ++i) {
      // Build a min-heap, the heap element is pair of (value, idx)
      // the top of the heap is the smallest value
      std::priority_queue<
          std::pair<T, TIndex>,
          std::vector<std::pair<T, TIndex>>,
          ValueCmp<T>>
          PQ;

      // Maintain the size of heap to be less or equal to k_, so the
      // heap will hold the k_ largest values
      for (TIndex j = 0; j < linear_shape[1]; ++j) {
        const auto value = input_map(i, j);
        if (PQ.size() < k_ || value > PQ.top().first) {
          PQ.push(std::make_pair(value, j));
        }
        if (PQ.size() > k_) {
          PQ.pop();
        }
      }
      for (TIndex j = 0; j < k_; ++j) {
        auto& pqElem = PQ.top();
        values_map(i, k_ - j - 1) = pqElem.first;
        indices_map(i, k_ - j - 1) = pqElem.second;
        PQ.pop();
      }
    }

    // Reshape output tensors to [a_1, a_2, ..., a_n, k]
    auto out_dims = in_dims;
    out_dims[out_dims.size() - 1] = k_;
    values->Reshape(out_dims);
    indices->Reshape(out_dims);
    return true;
  }

 private:
  int k_;
};

template <typename T, class Context>
class TopKGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  TopKGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& values = Input(0);
    auto& indices = Input(1);
    auto& original_input = Input(2);
    auto* output = Output(0);

    vector<TIndex> in_dims = values.dims();
    // Linearize input tensor except for last dimension
    // e.g. [3, 4, 5] -> [12, 5]
    // [5] -> [5]
    vector<TIndex> linear_shape = {size_to_dim_(in_dims.size() - 1, in_dims),
                                   in_dims[in_dims.size() - 1]};
    auto values_map = ConstEigenMatrixMapRowMajor<T>(
        static_cast<const T*>(values.raw_data()),
        linear_shape[0],
        linear_shape[1]);
    auto indices_map = ConstEigenMatrixMapRowMajor<TIndex>(
        static_cast<const TIndex*>(indices.raw_data()),
        linear_shape[0],
        linear_shape[1]);

    // Resize output tensors to be as orignial_input size and initialized with 0
    vector<TIndex> original_dims = original_input.dims();
    output->Resize(original_dims);
    T* output_data = output->template mutable_data<T>();
    memset(output_data, 0, output->nbytes());

    // Use Eigen maps to allow indexing into the 2d tensors
    auto output_map = EigenMatrixMapRowMajor<T>(
        output_data, linear_shape[0], original_dims.back());

    for (TIndex i = 0; i < linear_shape[0]; ++i) {
      for (TIndex j = 0; j < linear_shape[1]; ++j) {
        output_map(i, indices_map(i, j)) = values_map(i, j);
      }
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_TOP_K_H_
