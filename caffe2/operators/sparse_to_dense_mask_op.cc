#include "caffe2/operators/sparse_to_dense_mask_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(SparseToDenseMask, SparseToDenseMaskOp<CPUContext>);
REGISTER_CPU_OPERATOR(
    SparseToDenseMaskGradient,
    SparseToDenseMaskGradientOp<CPUContext>);

OPERATOR_SCHEMA(SparseToDenseMask)
    .NumInputs(3, 4)
    .NumOutputs(1, 2)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto mask = helper.template GetRepeatedArgument<int64_t>("mask");
      bool return_presence_mask = helper.template GetSingleArgument<bool>(
          "return_presence_mask", false);
      vector<TensorShape> out(1);

      if (in.size() == 4) {
        out[0].add_dims(in[3].dims(0));
      }
      out[0].add_dims(mask.size());
      for (const auto dim : in[2].dims()) {
        out[0].add_dims(dim);
      }
      out[0].set_data_type(in[2].data_type());

      if (return_presence_mask) {
        out.emplace_back();
        if (in.size() == 4) {
          out[1].add_dims(in[3].dims(0));
        }
        out[1].add_dims(mask.size());
        out[1].set_data_type(TensorProto::BOOL);
      }

      return out;
    })
    .SetDoc(R"DOC(
Convert sparse representations to dense with given indices.

Transforms a sparse representation of map<id, value> represented as `indices`
vector and `values` tensor into a compacted tensor where the first dimension
corresponds to each id provided in mask argument. Missing values are filled with
the value of `default_value`. After running this op:

  output[j, :] = values[i] # where mask[j] == indices[i]
  output[j, ...] = default_value # when mask[j] doesn't appear in indices

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
    .Arg(
        "return_presence_mask",
        "bool whether to return presence mask, false by default")
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
        "first dimension is omitted)")
    .Output(
        1,
        "presence_mask",
        "Bool tensor of shape `[len(lengths), len(mask)]` (if `lengths` is not "
        "provided the first dimension is omitted). True when a value for given "
        "id was present, false otherwise.");

OPERATOR_SCHEMA(SparseToDenseMaskGradient)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
The output is the gradient of the input value from SparseToDenseMask. The
gradient for default_value has not been implemented.
)DOC");

class GetSparseToDenseMaskGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> blob_names{I(0), GO(0)};

    // Add lengths blob if given
    if (def_.input_size() == 4) {
      blob_names.push_back(I(3));
    }
    return SingleGradientDef(
        "SparseToDenseMaskGradient", "", blob_names, vector<string>{GI(1)});
  }
};
REGISTER_GRADIENT(SparseToDenseMask, GetSparseToDenseMaskGradient);
} // namespace
} // namespace caffe2
