#include <algorithm>
#include <unordered_map>
#include <vector>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

class SparseToDenseMaskOp : public Operator<CPUContext> {
 public:
  SparseToDenseMaskOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    std::vector<int> mask = GetRepeatedArgument<int>("mask");
    featuresCount_ = mask.size();
    CAFFE_ENFORCE(!mask.empty(), "mask can't be empty");
    auto biggest = *std::max_element(mask.begin(), mask.end());
    dense_.assign(std::min(kMaxDenseSize, biggest + 1), -1);
    for (int i = 0; i < mask.size(); i++) {
      int id = mask[i];
      CAFFE_ENFORCE_GE(id, 0, "Only positive IDs are allowed.");
      if (id >= kMaxDenseSize) {
        CAFFE_ENFORCE(sparse_.count(id) == 0, "Duplicated id: ", id);
        sparse_[id] = i;
      } else {
        CAFFE_ENFORCE(dense_[id] == -1, "Duplicated id: ", id);
        dense_[id] = i;
      }
    }
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename TInd>
  bool DoRunWithType() {
    auto& sparse_indices = Input(INDICES);
    CAFFE_ENFORCE_EQ(sparse_indices.ndim(), 1);
    auto& sparse_values = Input(VALUES);
    CAFFE_ENFORCE_GE(sparse_values.ndim(), 1);
    CAFFE_ENFORCE_EQ(sparse_indices.size(), sparse_values.dim(0));
    auto& default_value = Input(DEFAULT);
    CAFFE_ENFORCE_EQ(default_value.ndim() + 1, sparse_values.ndim());
    CAFFE_ENFORCE_EQ(default_value.size(), sparse_values.size_from_dim(1));
    CAFFE_ENFORCE(sparse_values.meta() == default_value.meta());

    const TInd* sparse_indices_vec = sparse_indices.data<TInd>();
    const char* sparse_values_vec =
        static_cast<const char*>(sparse_values.raw_data());
    const void* default_val = default_value.raw_data();

    TIndex block_size = default_value.size();
    size_t block_nbytes = default_value.nbytes();

    int cols = featuresCount_;
    int rows = -1;
    int32_t default_length = sparse_indices.dim32(0);
    const int32_t* lengths_vec = nullptr;
    auto* output = Output(0);
    vector<TIndex> shape;
    if (InputSize() == 4) {
      auto& lengths = Input(LENGTHS);
      CAFFE_ENFORCE_EQ(lengths.ndim(), 1);
      lengths_vec = lengths.data<int32_t>();
      rows = lengths.dim32(0);
    }
    if (rows == -1) {
      // if the LENGTHS is not set, the output will be a vector
      rows = 1;
      lengths_vec = &default_length;
    } else {
      shape.push_back(rows);
    }
    shape.push_back(cols);
    shape.insert(
        shape.end(), default_value.dims().begin(), default_value.dims().end());
    output->Resize(shape);

    // init
    // TODO: consider unrolling CopyItems to make elemental types copy faster
    char* output_data =
        static_cast<char*>(output->raw_mutable_data(sparse_values.meta()));
    for (int i = 0; i < cols * rows; i++) {
      context_.template CopyItems<CPUContext, CPUContext>(
          default_value.meta(),
          block_size,
          default_val,
          output_data + i * block_nbytes);
    }

    int32_t offset = 0;
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < lengths_vec[r]; c++) {
        int idx = getFeatureIdx(sparse_indices_vec[offset + c]);
        if (idx != -1) {
          context_.template CopyItems<CPUContext, CPUContext>(
              sparse_values.meta(),
              block_size,
              sparse_values_vec + (offset + c) * block_nbytes,
              output_data + (r * cols + idx) * block_nbytes);
        }
      }
      offset += lengths_vec[r];
    }

    return true;
  }

 private:
  const int kMaxDenseSize = 1024 * 128;

  std::unordered_map<int, int> sparse_;
  std::vector<int> dense_;
  int featuresCount_;

  inline int getFeatureIdx(int id) const {
    if (id >= kMaxDenseSize) {
      const auto& iter = sparse_.find(id);
      if (iter == sparse_.end()) {
        return -1;
      } else {
        return iter->second;
      }
    } else {
      return (id >= dense_.size()) ? -1 : dense_[id];
    }
  }

  INPUT_TAGS(INDICES, VALUES, DEFAULT, LENGTHS);
};

namespace {
REGISTER_CPU_OPERATOR(SparseToDenseMask, SparseToDenseMaskOp);

OPERATOR_SCHEMA(SparseToDenseMask)
    .NumInputs(3, 4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Convert sparse representations to dense with given indices.

Transforms a sparse representation of map<id, value> represented as `indices`
vector and `values` tensor into a compacted tensor where the first dimension
corresponds to each id provided in mask argument. Missing values are filled with
the value of `default_value`. After running this op:

```
output[j, :] = values[i] # where mask[j] == indices[i]
output[j, ...] = default_value # when mask[j] doesn't appear in indices
```

If `lengths` is provided and not empty, and extra "batch" dimension is prepended
to the output.

`values` and `default_value` can have additional matching dimensions, operation
is performed on the entire subtensor in thise case.

For example, if `lengths` is supplied and `values` is 1-D vector of floats and
`default_value` is a float scalar, the output is going to be a float matrix
of size `len(lengths) X len(mask)`
)DOC")
    .Arg(
        "mask",
        "list(int) argument with desired ids on the 'dense' output dimension")
    .Input(0, "indices", "1-D int32/int64 tensor of concatenated ids of data")
    .Input(1, "values", "Data tensor, first dimension has to match `indices`")
    .Input(
        2,
        "default_value",
        "Default value for the output if the id is not present in `indices`. "
        "Must have the same type as `values` and the same shape, but without "
        "the first dimension")
    .Input(
        3,
        "lengths",
        "Optional lengths to represent a batch of `indices` and `values`.")
    .Output(
        0,
        "output",
        "Output tensor of the same type as `values` of shape `[len(lengths), "
        "len(mask)] + shape(default_value)` (if `lengths` is not provided the "
        "first dimension is omitted)");

NO_GRADIENT(SparseToDenseMask);
} // namespace
} // namespace caffe2
