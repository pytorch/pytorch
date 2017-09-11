#include "caffe2/operators/one_hot_ops.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename T>
bool BatchOneHotOp<CPUContext>::DoBatchOneHotOp(
    const TIndex batch_size,
    const TIndex in_dim,
    const Tensor<CPUContext>& input,
    const Tensor<CPUContext>& lens,
    const Tensor<CPUContext>& vals,
    Tensor<CPUContext>* output) {
  const auto* input_data = input.template data<T>();
  const auto* lens_data = lens.template data<int32_t>();
  const auto* vals_data = vals.template data<T>();

  const TIndex out_dim = vals.size();
  TIndex len_sum = 0;
  for (TIndex i = 0; i < in_dim; i++) {
    CAFFE_ENFORCE_GE(lens_data[i], 0);
    len_sum += lens_data[i];
  }
  CAFFE_ENFORCE_EQ(out_dim, len_sum);

  auto* output_data = output->template mutable_data<T>();
  memset(output_data, 0, output->nbytes());
  // eigen is column-major
  auto input_m = ConstEigenMatrixMap<T>(input_data, in_dim, batch_size);
  auto output_m = EigenMatrixMap<T>(output_data, out_dim, batch_size);

  // `p` is the column position in output_data, that points to the next
  // column to be filled.
  TIndex p = 0;
  // one-hot encoding for each example.
  for (TIndex j = 0; j < in_dim; j++) {
    for (TIndex t = 0; t < lens_data[j]; t++) {
      output_m.row(p) =
          input_m.row(j).cwiseEqual(vals_data[p]).template cast<T>();
      p++;
    }
  }
  return true;
}

template <>
void OneHotOp<CPUContext>::DoOneHotOp(
    TIndex batch_size,
    TIndex index_size,
    const Tensor<CPUContext>& indices,
    Tensor<CPUContext>* one_hots) {
  const TIndex* indices_ptr = indices.template data<TIndex>();
  float* one_hots_ptr = one_hots->template mutable_data<float>();
  memset(one_hots_ptr, 0, one_hots->nbytes());
  for (int i = 0; i < batch_size; ++i) {
    auto label_idx = indices_ptr[i];
    DCHECK((0 <= label_idx) && (label_idx < index_size));
    one_hots_ptr[label_idx] = 1.0;
    one_hots_ptr += index_size;
  }
}

class SegmentOneHotOp : public Operator<CPUContext> {
 public:
  SegmentOneHotOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& lengths = Input(0);
    auto& indices = Input(1);
    auto& index_size_tensor = Input(2);
    CAFFE_ENFORCE(lengths.ndim() == 1);
    CAFFE_ENFORCE(indices.ndim() == 1);
    CAFFE_ENFORCE(index_size_tensor.size() == 1);
    auto batch_size = lengths.size();
    auto index_size = *index_size_tensor.data<int64_t>();
    CAFFE_ENFORCE(index_size > 0);

    auto* lengths_ptr = lengths.data<int32_t>();
    auto* indices_ptr = indices.data<int64_t>();
    auto* one_hots = Output(0);
    one_hots->Resize(batch_size, index_size);
    auto* one_hots_ptr = one_hots->mutable_data<float>();
    if (one_hots->size() == 0) {
      return true;
    }
    memset(one_hots_ptr, 0, one_hots->nbytes());
    int el_idx = 0;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < lengths_ptr[i]; ++j) {
        DCHECK(el_idx < indices.size());
        auto label_idx = indices_ptr[el_idx++];
        DCHECK((0 <= label_idx) && (label_idx < index_size));
        one_hots_ptr[label_idx] = 1.0;
      }
      one_hots_ptr += index_size;
    }
    return true;
  }
};
REGISTER_CPU_OPERATOR(BatchOneHot, BatchOneHotOp<CPUContext>);
REGISTER_CPU_OPERATOR(OneHot, OneHotOp<CPUContext>);
REGISTER_CPU_OPERATOR(SegmentOneHot, SegmentOneHotOp);

OPERATOR_SCHEMA(BatchOneHot)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(Input is a matrix tensor. Its first dimension is the batch
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
        "output matrix that expands each input column with one hot encoding");

OPERATOR_SCHEMA(OneHot)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a sequence of indices, one for each example in a batch, returns a matrix
where each inner dimension has the size of the index and has 1.0 in the index
active in the given example, and 0.0 everywhere else.
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
}
