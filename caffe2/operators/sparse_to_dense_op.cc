#include "sparse_to_dense_op.h"

#include "caffe2/core/context.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SparseToDense, SparseToDenseOp<CPUContext>);

OPERATOR_SCHEMA(SparseToDense)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      if (in.size() == 3) {
        out[0].add_dims(in[2].dims(0));
      } else {
        out[0].set_unknown_shape(true);
        return out;
      }
      for (int i = 1; i < in[1].dims().size(); i++) {
        out[0].add_dims(in[1].dims(i));
      }
      out[0].set_data_type(in[1].data_type());
      return out;
    })
    .SetDoc(R"DOC(
Convert sparse representations to dense with given indices.

Transforms a sparse representation of map<id, value> represented as `indices`
vector and `values` tensor into a compacted tensor where the first dimension
is determined by the first dimension of the 3rd input if it is given or the
max index. Missing values are filled with zeros.

The op supports duplicated indices and performs summation over corresponding
values. This behavior is useful for converting GradientSlices into dense
representation.

After running this op:

  output[indices[i], :] += values[i]  // sum over all indices[i] equal to the index
  output[j, ...] = 0 if j not in indices
)DOC")
    .Input(0, "indices", "1-D int32/int64 tensor of concatenated ids of data")
    .Input(
        1,
        "values",
        "Data tensor, first dimension has to match `indices`, "
        "basic numeric types are supported")
    .Input(
        2,
        "data_to_infer_dim",
        "Optional: if provided, the first dimension of output is the first "
        "dimension of this tensor.")
    .Output(
        0,
        "output",
        "Output tensor of the same type as `values` of shape `[len(lengths), "
        "len(mask)] + shape(default_value)` (if `lengths` is not provided the "
        "first dimension is omitted)");


namespace {
class GetSparseToDenseGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Gather", "", vector<string>{GO(0), I(0)}, vector<string>{GI(1)});
  }
};

REGISTER_GRADIENT(SparseToDense, GetSparseToDenseGradient);
}
} // namespace caffe2
