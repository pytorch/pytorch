#include "caffe2/operators/flexible_top_k.h"

#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

namespace {

template <typename T>
struct ValueCmp {
  bool operator()(
      const std::pair<T, int64_t>& lhs,
      const std::pair<T, int64_t>& rhs) {
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

  const T* input_data = input.template data<T>();
  const int64_t* k_data = k.template data<int64_t>();

  // get flatten shape of input
  CAFFE_ENFORCE_GT(input.dim(), 0);
  vector<int64_t> input_dims = input.sizes().vec();
  vector<int64_t> linear_shape = {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      size_to_dim_(input_dims.size() - 1, input_dims), input_dims.back()};
  CAFFE_ENFORCE_EQ(
      linear_shape[0],
      k.numel(),
      "first n-1 dims of input data and K does not match.");

  int64_t output_size = 0;
  for (int64_t i = 0; i < linear_shape[0]; ++i) {
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
  auto* values = Output(0, {output_size}, at::dtype<T>());
  auto* indices = Output(1, {output_size}, at::dtype<int64_t>());
  T* values_data = values->template mutable_data<T>();
  int64_t* indices_data = indices->template mutable_data<int64_t>();

  int64_t output_offset = 0;
  // Sort preserving indices
  for (int64_t i = 0; i < linear_shape[0]; ++i) {
    // Build a min-heap, the heap element is pair of (value, idx)
    // the top of the heap is the smallest value
    std::priority_queue<
        std::pair<T, int64_t>,
        std::vector<std::pair<T, int64_t>>,
        ValueCmp<T>>
        PQ;

    int64_t k_ = k_data[i];
    for (int64_t j = 0; j < linear_shape[1]; ++j) {
      const T value = input_data[i * linear_shape[1] + j];
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      if (PQ.size() < k_ || value > PQ.top().first) {
        PQ.push(std::make_pair(value, j));
      }
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      if (PQ.size() > k_) {
        PQ.pop();
      }
    }
    for (int64_t j = 0; j < k_; ++j) {
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

  const int64_t* k_data = k.template data<int64_t>();
  const T* values_data = values.template data<T>();
  const int64_t* indices_data = indices.template data<int64_t>();

  // Resize output tensors to be as orignial_input size and initialized with 0
  CAFFE_ENFORCE_GT(original_input.dim(), 0);
  vector<int64_t> original_dims = original_input.sizes().vec();
  auto* output = Output(0, original_dims, at::dtype<T>());
  T* output_data = output->template mutable_data<T>();
  math::Set<T, Context>(
      output->numel(), static_cast<T>(0), output_data, &context_);

  int64_t index_offset = 0;
  for (int64_t i = 0; i < k.numel(); ++i) {
    // offset of output_data
    int64_t output_offset = i * original_dims.back();
    for (int64_t j = 0; j < k_data[i]; ++j) {
      int64_t index = indices_data[index_offset + j];
      T value = values_data[index_offset + j];
      output_data[output_offset + index] = value;
    }
    index_offset += k_data[i];
  }

  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(FlexibleTopK, FlexibleTopKOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    FlexibleTopKGradient,
    FlexibleTopKGradientOp<float, CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(FlexibleTopK, GetFlexibleTopKGradient);

} // namespace caffe2
