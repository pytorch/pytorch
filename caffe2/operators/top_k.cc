#include "caffe2/operators/top_k.h"

#include "caffe2/proto/caffe2.pb.h"

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
bool TopKOp<T, Context>::RunOnDevice() {
  auto& input = Input(0);
  auto* values = Output(0);
  auto* indices = Output(1);
  auto* flatten_indices = OutputSize() > 2 ? Output(2) : nullptr;

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
  if (flatten_indices) {
    flatten_indices->Resize(linear_shape[0] * k_);
  }

  // Use Eigen maps to allow indexing into the 2d tensors like values_map(i,j)
  auto values_map = EigenMatrixMapRowMajor<T>(
      values->template mutable_data<T>(), linear_shape[0], k_);
  auto indices_map = EigenMatrixMapRowMajor<TIndex>(
      indices->template mutable_data<TIndex>(), linear_shape[0], k_);
  auto* flatten_indices_data = flatten_indices
      ? flatten_indices->template mutable_data<TIndex>()
      : nullptr;

  TIndex flatten_offset = 0;
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
      if (flatten_indices_data) {
        flatten_indices_data[k_ - j - 1] = pqElem.second + flatten_offset;
      }
      PQ.pop();
    }
    if (flatten_indices_data) {
      flatten_indices_data += k_;
    }
    flatten_offset += linear_shape[1];
  }

  // Reshape output tensors to [a_1, a_2, ..., a_n, k]
  auto out_dims = in_dims;
  out_dims[out_dims.size() - 1] = k_;
  values->Reshape(out_dims);
  indices->Reshape(out_dims);
  return true;
}

template <typename T, class Context>
bool TopKGradientOp<T, Context>::RunOnDevice() {
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

namespace {

REGISTER_CPU_OPERATOR(TopK, TopKOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(TopKGradient, TopKGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(TopK)
    .NumInputs(1)
    .NumOutputs(2, 3)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out = {in[0], in[0]};
      ArgumentHelper helper(def);
      auto k = helper.GetSingleArgument("k", -1);
      auto dims_size = in[0].dims_size();
      out[0].set_dims(dims_size - 1, k);
      out[1].set_dims(dims_size - 1, k);
      out[1].set_data_type(TensorProto_DataType_INT32);
      if (def.output_size() > 2) {
        TensorShape flatten_indices_shape;
        flatten_indices_shape.set_data_type(TensorProto_DataType_INT32);
        flatten_indices_shape.add_dims(
            std::accumulate(
                in[0].dims().begin(),
                in[0].dims().end() - 1,
                1,
                std::multiplies<long>()) *
            k);
        out.push_back(flatten_indices_shape);
      }
      return out;
    })
    .SetDoc(R"DOC(
Retrieve the top-K elements for the last dimension. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
-Value tensor of shape [a_1, a_2, ..., a_n, k] which contains the values of
 the top k elements along the last dimension
-Index tensor of shape [a_1, a_2, ..., a_n, k] which contains the indices
 of the top k elements (original indices from the input tensor).

Given two equivalent values, this operator uses the indices along the last dim-
ension as a tiebreaker. That is, the element with the lower index will appear
first.
    )DOC")
    .Input(0, "X", "Tensor of shape [a_1, a_2, ..., a_n, r]")
    .Output(
        0,
        "Values",
        "Tensor of shape [a_1, a_2, ..., a_n, k] containing"
        " top K values from the input tensor")
    .Output(
        1,
        "Indices",
        "Tensor of shape [a_1, a_2, ..., a_n, k] containing"
        " the corresponding input tensor indices for the top K values.")
    .Output(
        2,
        "Flatten indices",
        "Tensor of shape [a_1 * a_2 * ... * a_n * k] containing the indices "
        "into the flatten input")
    .Arg("k", "Number of top elements to retrieve");

OPERATOR_SCHEMA(TopKGradient).NumInputs(3).NumOutputs(1);

class GetTopKGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "TopKGradient",
        "",
        vector<string>{GO(0), O(1), I(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(TopK, GetTopKGradient);

} // namespace
} // namespace caffe2
