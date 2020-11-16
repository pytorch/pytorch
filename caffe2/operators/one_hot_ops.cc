#include "caffe2/operators/one_hot_ops.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename T>
bool BatchOneHotOp<CPUContext>::DoRunWithType() {
  auto& input = Input(X);
  auto& lens = Input(LENS);
  auto& vals = Input(VALS);
  CAFFE_ENFORCE_GE(input.dim(), 1);
  auto N = input.size(0);
  auto D = input.size_from_dim(1);
  CAFFE_ENFORCE_EQ(lens.numel(), D);

  const auto* lens_data = lens.template data<int32_t>();
  int64_t output_dim = 0;
  valsOffsets_.resize(D + 1);
  for (int64_t i = 0; i < D; i++) {
    CAFFE_ENFORCE_GE(lens_data[i], 0);
    valsOffsets_[i] = output_dim;
    output_dim += lens_data[i];
  }
  valsOffsets_[D] = output_dim;

  CAFFE_ENFORCE_EQ(vals.numel(), output_dim);

  auto* output = Output(ONE_HOT, {N, output_dim}, at::dtype<T>());

  const auto* input_data = input.template data<T>();
  const auto* vals_data = vals.template data<T>();
  auto* output_data = output->template mutable_data<T>();

  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < D; j++) {
      const auto input_val = input_data[i * D + j];
      for (int64_t k = valsOffsets_[j]; k < valsOffsets_[j + 1]; ++k) {
        output_data[k] = vals_data[k] == input_val;
      }
    }
    output_data += output_dim;
  }

  return true;
}

vector<TensorShape> TensorInferenceForBatchOneHot(
    const OperatorDef& /* def */,
    const vector<TensorShape>& in) {
  std::vector<int64_t> output_dims(2);
  output_dims[0] = in[0].dims(0); // N
  output_dims[1] = in[2].dims(0); // vals.size()
  return vector<TensorShape>{
      CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
}

vector<TensorShape> TensorInferenceForBucketBatchOneHot(
    const OperatorDef& /* def */,
    const vector<TensorShape>& in) {
  std::vector<int64_t> output_dims(2);
  output_dims[0] = in[0].dims(0); // N
  output_dims[1] = in[1].dims(0) + in[2].dims(0); // vals.size() + length.size()
  return vector<TensorShape>{
      CreateTensorShape(vector<int64_t>{output_dims}, in[0].data_type())};
}

OpSchema::Cost CostInferenceForBatchOneHot(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  CAFFE_ENFORCE_EQ(in.size(), 3, "BatchOneHot requires three inputs");
  struct OpSchema::Cost c;
  const TensorShape output = TensorInferenceForBatchOneHot(def, in)[0];

  const auto& data = in[0];
  const auto& length = in[1];
  const auto& values = in[2];

  uint64_t nBytesData = nElemFromDim(data) * sizeof(data.data_type());
  uint64_t nBytesLength = nElemFromDim(length) * sizeof(length.data_type());
  uint64_t nBytesValues = nElemFromDim(values) * sizeof(values.data_type());
  c.flops = 0;
  c.bytes_read = nBytesData + nBytesLength + nBytesValues;
  c.bytes_written = nElemFromDim(output) * sizeof(output.data_type());
  c.params_bytes = 0;
  return c;
}

template <>
void OneHotOp<CPUContext>::DoOneHotOp(
    int64_t batch_size,
    int64_t index_size,
    const Tensor& indices,
    Tensor* one_hots) {
  const int64_t* indices_ptr = indices.template data<int64_t>();
  float* one_hots_ptr = one_hots->template mutable_data<float>();
  memset(one_hots_ptr, 0, one_hots->nbytes());
  for (int i = 0; i < batch_size; ++i) {
    auto label_idx = indices_ptr[i];
    DCHECK((0 <= label_idx) && (label_idx < index_size));
    one_hots_ptr[label_idx] = 1.0;
    one_hots_ptr += index_size;
  }
}

template <>
bool BatchBucketOneHotOp<CPUContext>::RunOnDevice() {
  auto& input = Input(X);
  auto& lens = Input(LENS);
  auto& boundaries = Input(BOUNDARIES);
  CAFFE_ENFORCE_GE(input.dim(), 1);
  auto N = input.size(0);
  auto D = input.size_from_dim(1);
  CAFFE_ENFORCE_EQ(lens.numel(), D);

  const auto* lens_data = lens.template data<int32_t>();

  CAFFE_ENFORCE_EQ(
      std::accumulate(lens_data, lens_data + lens.numel(), 0),
      boundaries.numel(),
      "The sum of length should be equal to the length of boundaries");

  int64_t output_dim = 0;
  for (int64_t i = 0; i < D; i++) {
    CAFFE_ENFORCE_GT(lens_data[i], 0);
    // Number of buckets is number of bucket edges + 1
    output_dim += (lens_data[i] + 1);
  }

  auto* output = Output(ONE_HOT, {N, output_dim}, at::dtype<float>());

  const auto* input_data = input.template data<float>();
  const auto* boundaries_data = boundaries.template data<float>();
  auto* output_data = output->template mutable_data<float>();

  math::Set<float, CPUContext>(output->numel(), 0.f, output_data, &context_);

  int64_t pos = 0;
  for (int64_t i = 0; i < N; i++) {
    auto* boundaries_offset = boundaries_data;
    int64_t output_offset = 0;

    for (int64_t j = 0; j < D; j++) {
      // here we assume the boundary values for each feature are sorted
      int64_t lower_bucket_idx = std::lower_bound(
                                    boundaries_offset,
                                    boundaries_offset + lens_data[j],
                                    input_data[pos]) -
          boundaries_offset;

      int64_t upper_bucket_idx = std::upper_bound(
                                    boundaries_offset,
                                    boundaries_offset + lens_data[j],
                                    input_data[pos]) -
          boundaries_offset;

      int64_t bucket_idx = (lower_bucket_idx + upper_bucket_idx) / 2;
      output_data[i * output_dim + output_offset + bucket_idx] = 1.0;
      boundaries_offset += lens_data[j];
      output_offset += (lens_data[j] + 1);
      pos++;
    }
  }

  return true;
};

class SegmentOneHotOp : public Operator<CPUContext> {
 public:
  template <class... Args>
  explicit SegmentOneHotOp(Args&&... args)
      : Operator(std::forward<Args>(args)...) {}

  bool RunOnDevice() override {
    auto& lengths = Input(0);
    auto& indices = Input(1);
    auto& index_size_tensor = Input(2);
    CAFFE_ENFORCE(lengths.dim() == 1);
    CAFFE_ENFORCE(indices.dim() == 1);
    CAFFE_ENFORCE(index_size_tensor.numel() == 1);
    auto batch_size = lengths.numel();
    auto index_size = *index_size_tensor.data<int64_t>();
    CAFFE_ENFORCE(index_size > 0);

    auto* lengths_ptr = lengths.data<int32_t>();
    auto* indices_ptr = indices.data<int64_t>();

    auto* one_hots = Output(0, {batch_size, index_size}, at::dtype<float>());
    auto* one_hots_ptr = one_hots->template mutable_data<float>();
    if (one_hots->numel() == 0) {
      return true;
    }
    memset(one_hots_ptr, 0, one_hots->nbytes());
    int el_idx = 0;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < lengths_ptr[i]; ++j) {
        DCHECK(el_idx < indices.numel());
        auto label_idx = indices_ptr[el_idx++];
        DCHECK((0 <= label_idx) && (label_idx < index_size));
        one_hots_ptr[label_idx] = 1.0;
      }
      one_hots_ptr += index_size;
    }
    return true;
  }
};
REGISTER_CPU_OPERATOR(BatchBucketOneHot, BatchBucketOneHotOp<CPUContext>);
REGISTER_CPU_OPERATOR(BatchOneHot, BatchOneHotOp<CPUContext>);
REGISTER_CPU_OPERATOR(OneHot, OneHotOp<CPUContext>);
REGISTER_CPU_OPERATOR(SegmentOneHot, SegmentOneHotOp);

OPERATOR_SCHEMA(BatchBucketOneHot)
    .NumInputs(3)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable the filler
    .SetDoc(R"DOC(
Input is a matrix tensor. Its first dimension is the batch
size. For each column, bucketize it based on the boundary values and then do
one hot encoding. The `lengths` specifies the number of boundary values for each
column. The final number of buckets is this number plus 1. This would also be
the expanded feature size. `boundaries` specifies all the boundary values.
Note that each bucket is right-inclusive. That is, given boundary values
[b1, b2, b3], the buckets are defined as (-int, b1], (b1, b2], (b2, b3], (b3, inf).
For example

  data = [[2, 3], [4, 1], [2, 5]], lengths = [2, 3],
  If boundaries = [0.1, 2.5, 1, 3.1, 4.5], then
  output = [[0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1]]

  If boundaries = [0.1, 2.5, 1, 1, 3.1], then
  output = [[0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 1]]

)DOC")
    .Input(0, "data", "input tensor matrix")
    .Input(1, "lengths", "the size is the same as the width of the `data`")
    .Input(2, "boundaries", "bucket boundaries")
    .Output(
        0,
        "output",
        "output matrix that expands each input column with one hot encoding"
        "based on the bucketization")
    .TensorInferenceFunction(TensorInferenceForBucketBatchOneHot);

OPERATOR_SCHEMA(BatchOneHot)
    .NumInputs(3)
    .NumOutputs(1)
    .ValueKeyLengthInputFillers(
        BatchOneHotOp<CPUContext>::X,
        BatchOneHotOp<CPUContext>::VALS,
        BatchOneHotOp<CPUContext>::LENS)
    .SetDoc(R"DOC(
Input is a matrix tensor. Its first dimension is the batch
size. Expand each column of it using one hot encoding. The `lengths` specifies
the size of each column after encoding, and the `values` is the dictionary value
of one-hot encoding for each column. For example

  If data = [[2, 3], [4, 1], [2, 5]], lengths = [2, 3],
  and values = [2, 4, 1, 3, 5], then

  output = [[1, 0, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 0, 1]]
)DOC")
    .Input(0, "data", "input tensor matrix")
    .Input(1, "lengths", "the size is the same as the width of the `data`")
    .Input(2, "values", "one hot encoding dictionary values")
    .Output(
        0,
        "output",
        "output matrix that expands each input column with one hot encoding")
    .TensorInferenceFunction(TensorInferenceForBatchOneHot)
    .CostInferenceFunction(
        OpSchema::CostInferenceFunctionType(CostInferenceForBatchOneHot));

OPERATOR_SCHEMA(OneHot)
    .NumInputs(2)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable the filler
    .SetDoc(R"DOC(
The *OneHot* op accepts two inputs *indices* and *index_size_tensor*, and produces a single output *one_hots*.  For each index in *indices* the op creates a one-hot row in *one_hots* of length *index_size_tensor* where all entries are zero except the entry at the index is 1. The size of *one_hots* is *len(indices)* x *index_size_tensor*.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/one_hot_ops.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "OneHot",
    ["indices", "index_size_tensor"],
    ["one_hots"],
)

workspace.FeedBlob("indices", np.array([0,1,2,3,4]).astype(np.long))
print("indices:\n", workspace.FetchBlob("indices"))

workspace.FeedBlob("index_size_tensor", np.array([5]).astype(np.long))
print("index_size_tensor:\n", workspace.FetchBlob("index_size_tensor"))

workspace.RunOperatorOnce(op)
print("one_hots: \n", workspace.FetchBlob("one_hots"))

```

**Result**

```

indices:
 [0 1 2 3 4]
index_size_tensor:
 [5]
one_hots:
 [[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]

```

</details>

)DOC")
    .Input(0, "indices", "The active index for each example in the batch.")
    .Input(
        1,
        "index_size_tensor",
        "Scalar with the size of the index. Must be in CPU context")
    .Output(0, "one_hots", "Matrix of size len(indices) x index_size");

OPERATOR_SCHEMA(SegmentOneHot)
    .NumInputs(3)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable the filler
    .SetDoc(R"DOC(
Given a sequence of indices, segmented by the lengths tensor, returns a matrix
that has the elements in each sequence set to 1.0, and 0.0 everywhere else.
)DOC")
    .Input(0, "lengths", "Size of each segment.")
    .Input(1, "indices", "Active indices, of size sum(lengths)")
    .Input(2, "index_size_tensor", "Size of the index")
    .Output(0, "one_hots", "Matrix of size len(lengths) x index_size");

NO_GRADIENT(BatchOneHot);
NO_GRADIENT(OneHot);
NO_GRADIENT(SegmentOneHot);
NO_GRADIENT(BucketBatchOneHot);
} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    BatchBucketOneHot,
    "_caffe2::BatchBucketOneHot(Tensor data, Tensor lengths, Tensor boundaries) -> Tensor output",
    caffe2::BatchBucketOneHotOp<caffe2::CPUContext>);
