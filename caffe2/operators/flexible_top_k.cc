#include "caffe2/operators/flexible_top_k.h"

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

} // namespace

template <typename T, class Context>
bool FlexibleTopKOp<T, Context>::RunOnDevice() {
  auto& input = Input(0);
  auto& k = Input(1);
  auto* values = Output(0);
  auto* indices = Output(1);

  const T* input_data = input.template data<T>();
  const TIndex* k_data = k.template data<TIndex>();

  // get flatten shape of input
  CAFFE_ENFORCE_GT(input.ndim(), 0);
  vector<TIndex> input_dims = input.dims();
  vector<TIndex> linear_shape = {
      size_to_dim_(input_dims.size() - 1, input_dims), input_dims.back()};
  CAFFE_ENFORCE_EQ(
      linear_shape[0],
      k.size(),
      "first n-1 dims of input data and K does not match.");

  TIndex output_size = 0;
  for (TIndex i = 0; i < linear_shape[0]; ++i) {
    CAFFE_ENFORCE(
        linear_shape[1] >= k_data[i],
        "k should not be greater than last dim, error at index ",
        i,
        ", with value: ",
        k_data[i]);
    CAFFE_ENFORCE(
        k_data[i] > 0,
        "k should be greater than 0, error at index ",
        i,
        ",  with value: ",
        k_data[i]);
    output_size += k_data[i];
  }
  values->Resize(output_size);
  indices->Resize(output_size);
  T* values_data = values->template mutable_data<T>();
  TIndex* indices_data = indices->template mutable_data<TIndex>();

  TIndex output_offset = 0;
  // Sort preserving indices
  for (TIndex i = 0; i < linear_shape[0]; ++i) {
    // Build a min-heap, the heap element is pair of (value, idx)
    // the top of the heap is the smallest value
    std::priority_queue<
        std::pair<T, TIndex>,
        std::vector<std::pair<T, TIndex>>,
        ValueCmp<T>>
        PQ;

    TIndex k_ = k_data[i];
    for (TIndex j = 0; j < linear_shape[1]; ++j) {
      const T value = input_data[i * linear_shape[1] + j];
      if (PQ.size() < k_ || value > PQ.top().first) {
        PQ.push(std::make_pair(value, j));
      }
      if (PQ.size() > k_) {
        PQ.pop();
      }
    }
    for (TIndex j = 0; j < k_; ++j) {
      auto& pqElem = PQ.top();
      values_data[output_offset + k_ - j - 1] = pqElem.first;
      indices_data[output_offset + k_ - j - 1] = pqElem.second;
      PQ.pop();
    }
    output_offset += k_;
  }

  return true;
}

template <typename T, class Context>
bool FlexibleTopKGradientOp<T, Context>::RunOnDevice() {
  auto& original_input = Input(0);
  auto& k = Input(1);
  auto& values = Input(2);
  auto& indices = Input(3);
  auto* output = Output(0);

  const TIndex* k_data = k.template data<TIndex>();
  const T* values_data = values.template data<T>();
  const TIndex* indices_data = indices.template data<TIndex>();

  // Resize output tensors to be as orignial_input size and initialized with 0
  CAFFE_ENFORCE_GT(original_input.ndim(), 0);
  vector<TIndex> original_dims = original_input.dims();
  output->Resize(original_dims);
  T* output_data = output->template mutable_data<T>();
  math::Set<T, Context>(
      output->size(), static_cast<T>(0), output_data, &context_);

  TIndex index_offset = 0;
  for (TIndex i = 0; i < k.size(); ++i) {
    // offset of output_data
    TIndex output_offset = i * original_dims.back();
    for (TIndex j = 0; j < k_data[i]; ++j) {
      TIndex index = indices_data[index_offset + j];
      T value = values_data[index_offset + j];
      output_data[output_offset + index] = value;
    }
    index_offset += k_data[i];
  }

  return true;
}

REGISTER_CPU_OPERATOR(FlexibleTopK, FlexibleTopKOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    FlexibleTopKGradient,
    FlexibleTopKGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(FlexibleTopK)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Given two tensors: X and K,
retrieve the top K[..., 1] elements from X on the last dimension.
X is an input tensor of shape [a_1, a_2, ..., a_n, r].
K is an input tensor of shape [a_1, a_2, ..., a_n, 1],
where for each element, r >= K[..., 1] > 0
Output two outputs:
-Flatten values tensor of shape [ \sum_i K[i, 1] ] which contains the values of
 the top K[..., 1]  elements along the last dimension
-Flatten indices tensor of shape [ \sum_i K[i, 1] ] which contains the indices
 of the top K[..., 1]  elements, flatten indices from the input tensor).
These two outputs should be used with the input K, so that we know which indices
in X are picked.

Given two equivalent values, this operator uses the indices along the last dim-
ension as a tiebreaker. That is, the element with the lower index will appear
first.
    )DOC")
    .Input(0, "X", "Tensor of shape [a_1, a_2, ..., a_n, r]")
    .Input(1, "K", "Tensor of shape [a_1, a_2, ..., a_n, 1]")
    .Output(
        0,
        "Flatten values",
        "Tensor of shape [ \\sum_i K[i, 1] ] containing"
        " top K[..., 1] values from the input tensor")
    .Output(
        1,
        "Flatten indices",
        "Tensor of shape [ \\sum_i K[i, 1] ] containing the indices "
        "into the flatten input");

OPERATOR_SCHEMA(FlexibleTopKGradient).NumInputs(4).NumOutputs(1);

class GetFlexibleTopKGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "FlexibleTopKGradient",
        "",
        vector<string>{I(0), I(1), GO(0), O(1)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(FlexibleTopK, GetFlexibleTopKGradient);

} // namespace caffe2
