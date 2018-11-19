#include "batch_sparse_to_dense_op.h"

#include "caffe2/core/context.h"

namespace caffe2 {

template <typename T, class Context>
bool BatchSparseToDenseOp<T, Context>::RunOnDevice() {
  auto& lengths = Input(LENGTHS);
  auto& indices = Input(INDICES);
  auto& values = Input(VALUES);
  auto* output = Output(0);
  CAFFE_ENFORCE_EQ(indices.numel(), values.numel());
  CAFFE_ENFORCE_EQ(lengths.dim(), 1);
  CAFFE_ENFORCE_EQ(indices.dim(), 1);

  const int64_t* lengths_data = lengths.template data<int64_t>();
  const int64_t* indices_data = indices.template data<int64_t>();
  const T* values_data = values.template data<T>();
  int64_t batch_size = lengths.numel();
  int64_t lengths_sum = 0;
  math::Sum<int64_t, Context>(batch_size, lengths_data, &lengths_sum, &context_);
  CAFFE_ENFORCE_EQ(lengths_sum, indices.numel());

  vector<int64_t> output_shape = {batch_size};
  if (InputSize() == 4) {
    auto& shaper = Input(3);
    CAFFE_ENFORCE_EQ(shaper.dim(), 2);
    if (dense_last_dim_ == -1) {
      dense_last_dim_ = shaper.size(1);
    } else {
      CAFFE_ENFORCE(
          dense_last_dim_ == shaper.size(1),
          "The last dim argument is not aligned with the shape input last dim");
    }
  } else {
    CAFFE_ENFORCE(dense_last_dim_ >= 1, "The last dim of dense must be >= 1");
  }
  output_shape.push_back(dense_last_dim_);
  output->Resize(output_shape);
  T* output_data = output->template mutable_data<T>();
  math::Set(
      output->numel(), static_cast<T>(default_value_), output_data, &context_);

  int64_t k = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    for (int64_t j = 0; j < lengths_data[i]; ++j) {
      CAFFE_ENFORCE(
          indices_data[k] < dense_last_dim_,
          "An indice (",
          indices_data[k],
          ") is larger then last dim of dense (",
          dense_last_dim_,
          ").");
      output_data[i * dense_last_dim_ + indices_data[k]] = values_data[k];
      k += 1;
    }
  }

  return true;
}

template <typename T, class Context>
bool BatchDenseToSparseOp<T, Context>::RunOnDevice() {
  auto& lengths = Input(LENGTHS);
  auto& indices = Input(INDICES);
  auto& dense = Input(DENSE);
  auto* output = Output(0);
  CAFFE_ENFORCE_EQ(lengths.dim(), 1);
  CAFFE_ENFORCE_EQ(indices.dim(), 1);
  CAFFE_ENFORCE_EQ(dense.dim(), 2);
  const int64_t* lengths_data = lengths.template data<int64_t>();
  const int64_t* indices_data = indices.template data<int64_t>();
  const T* dense_data = dense.template data<T>();

  int64_t batch_size = lengths.numel();
  int64_t lengths_sum = 0;
  math::Sum<int64_t, Context>(batch_size, lengths_data, &lengths_sum, &context_);
  CAFFE_ENFORCE_EQ(lengths_sum, indices.numel());

  CAFFE_ENFORCE_EQ(batch_size, dense.size(0));
  dense_last_dim_ = dense.size(1);
  vector<int64_t> output_shape = indices.sizes().vec();
  output->Resize(output_shape);
  T* output_data = output->template mutable_data<T>();

  int64_t k = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    for (int64_t j = 0; j < lengths_data[i]; ++j) {
      CAFFE_ENFORCE(
          indices_data[k] < dense.size(1),
          "An indice (",
          indices_data[k],
          ") is larger then last dim of dense (",
          dense.size(1),
          ").");
      output_data[k] = dense_data[i * dense.size(1) + indices_data[k]];
      k += 1;
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(
    BatchSparseToDense,
    BatchSparseToDenseOp<float, CPUContext>);

OPERATOR_SCHEMA(BatchSparseToDense)
    .NumInputs(3, 4)
    .NumOutputs(1)
    .DisallowInputFillers() // TODO: enable the filler
    .SetDoc(R"DOC(
Convert sparse matrix representation into dense matrix.

A sparse matrix is represented by `lengths` vector, `indices` vector,
and `values` vector. Each element in `lengths` vector (lengths[`i`]) represents
the number of indices in this batch (batch `i`).
With in each batch, `indices` should not have duplicate number.

For example, with input:

  lengths = [2, 3, 1]
  indices = [0, 1, 2, 3, 4, 5]
  values =  [6, 7, 8, 9, 10, 11]
  dense_dim = 6
  default_value = 0

The output is:

  output = [[6, 7, 0, 0, 0,  0],
            [0, 0, 8, 9, 10, 0],
            [0, 0, 0, 0, 0, 11]]

after running this operator.
)DOC")
    .Input(
        0,
        "lengths",
        "Flatten tensor, used to break down indices and values into per batch indices and values.")
    .Input(
        1,
        "indices",
        "Flatten tensor of total size = \\sum lengths, containing the indices ")
    .Input(2, "values", "Data tensor, dimension has to match `indices`")
    .Input(
        3,
        "output_shape_inference",
        "Optional, a dense tensor whose shape define the output shape")
    .Output(
        0,
        "dense",
        "2-D dense tensor, with 1st dim = len(lengths), 2nd dim = dense_last_dim"
        "in the arg list, the tensor is of the same data type as `values`."
        "Missing values are filled with default_value")
    .Arg(
        "dense_last_dim",
        "Optional, output dense last dimension. "
        "If both this argument and output_shape_inference are set, "
        "it should be consistent with output_shape_inference's last dim")
    .Arg(
        "default_value",
        "Optional, missing values are filled with this value."
        "default_value = 0 when not set");

REGISTER_CPU_OPERATOR(
    BatchDenseToSparse,
    BatchDenseToSparseOp<float, CPUContext>);

OPERATOR_SCHEMA(BatchDenseToSparse)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This Op is a inverse of BatchSparseToDenseOp.
Basically, given a `lengths` vector, a `indices` vector,
and a dense matrix `dense`, output `value` vector so that, along with
`lengths` vector and `indices` vector, forms a sparse representation
of the dense matrix.

A sparse matrix is represented by `lengths` vector, `indices` vector,
and `values` vector. Each element in `lengths` vector (lengths[`i`]) represents
the number of indices in this batch (batch `i`).
With in each batch, `indices` should not have duplicate number.

For example, with input:

  lengths = [2, 3, 1]
  indices = [0, 1, 2, 3, 4, 5]
  output = [[6, 7, 0, 0, 0,  0],
            [0, 0, 8, 9, 10, 0],
            [0, 0, 0, 0, 0, 11]]

The output is:

  values = [6, 7, 8, 9, 10, 11]

after running this operator.
)DOC")
    .Input(
        0,
        "lengths",
        "Flatten lengths, Used to break down indices into per batch indices")
    .Input(
        1,
        "indices",
        "Flatten indices, tensor of total size = \\sum lengths, containing the indices ")
    .Input(
        2,
        "dense",
        "dense 2-D tensor, first dim = len(lengths), last dim > Any(indices)")
    .Output(
        0,
        "values",
        "Values, tensor of the same size as `indices` and same data type as dense tensor.");

namespace {

class GetBatchSparseToDenseGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "BatchDenseToSparse",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(2)});
  }
};

class GetBatchDenseToSparseGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "BatchSparseToDense",
        "",
        vector<string>{I(0), I(1), GO(0), I(2)},
        vector<string>{GI(2)});
  }
};

REGISTER_GRADIENT(BatchSparseToDense, GetBatchSparseToDenseGradient);
REGISTER_GRADIENT(BatchDenseToSparse, GetBatchDenseToSparseGradient);

} // namespace
} // namespace caffe2
