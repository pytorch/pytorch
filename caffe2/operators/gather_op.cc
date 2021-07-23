#include "gather_op.h"
namespace caffe2 {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Gather, GatherOp<CPUContext>);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Gather)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(

The *Gather* op accepts a *DATA* tensor of rank $r >= 1$ and *INDICES* tensor of rank $q$ as inputs. It then gathers entries of the outer-most dimension of *DATA*, indexed by *INDICES*, and concatenate them in an output tensor of rank $q + (r - 1)$.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.cc
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Gather",
    ["DATA", "INDICES"],
    ["OUTPUT"]
)
data = np.array([[1., 1.2],[2.3, 3.4],[4.5, 5.7]])
print("DATA:\n",data)

inds = np.array([[0, 1],[1, 2]])
print("INDICES:\n",inds)

// Feed X into workspace
workspace.FeedBlob("DATA", data.astype(np.float32))
workspace.FeedBlob("INDICES", inds.astype(np.int32))

workspace.RunOperatorOnce(op)
print("OUTPUT:\n", workspace.FetchBlob("OUTPUT"))

```

**Result**

```

DATA:
 [[1.  1.2]
 [2.3 3.4]
 [4.5 5.7]]
INDICES:
 [[0 1]
 [1 2]]
OUTPUT:
 [[[1.  1.2]
  [2.3 3.4]]

 [[2.3 3.4]
  [4.5 5.7]]]

```

</details>

)DOC")
    .Input(0, "DATA", "Input data tensor of rank $r>=1$")
    .Input(
        1,
        "INDICES",
        "Input indices tensor of rank $q$. This tensor must contain integers.")
    .Output(0, "OUTPUT", "Output tensor of rank $q+(r-1)$")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      const int axis = helper.GetSingleArgument<int>("axis", 0);
      const bool match_outer =
          helper.GetSingleArgument<bool>("match_outer", false);
      const auto& data_dims = GetDimsVector(in[0]);
      const auto& indices_dims = GetDimsVector(in[1]);

      vector<int> output_dims =
          caffe2::gather_helper::calc_output_shape_vector<int>(
              data_dims, indices_dims, axis, match_outer);
      vector<TensorShape> out(1);
      out[0] = CreateTensorShape(output_dims, in[0].data_type());
      return out;
    })
    .InheritOnnxSchema();

class GetGatherGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    const bool dense_gradient =
        argsHelper.GetSingleArgument<bool>("dense_gradient", false);
    const int axis = argsHelper.GetSingleArgument<int>("axis", 0);

    // TBD: While it hasn't been used yet, we need to add wrap_indices support
    // to gradients next.
    // if (argsHelper.HasArgument("wrap_indices_")) {
    // }

    using Op = GatherOp<CPUContext>;

    if (axis == 0) {
      if (dense_gradient) {
        return vector<OperatorDef>{CreateOperatorDef(
            "SparseToDense",
            "",
            vector<string>{I(Op::INDICES), GO(0), I(Op::DATA)},
            vector<string>{GI(Op::DATA)})};
      } else {
        // For now we don't do any reshaping as the consumer of this op would
        // probably be ScatterUpdate which is intenionally ignores shapes. We
        // might need to revisit it in the future for correctness purposes. The
        // right shape for the output woild be to flatten INDICES and collapse
        // first X dims of GRAD
        SetSparse(Op::DATA, I(Op::INDICES), GO(0));
        return vector<OperatorDef>();
      }
    }

    // TBD: This is misleading to use dense_gradient by default for axis 0
    // and not othewise....
    if (argsHelper.HasArgument("dense_gradient")) {
      CAFFE_ENFORCE(
          dense_gradient == true,
          "Gather with axis > 0 must use dense_gradient");
    }

    Argument axisArg = MakeArgument<int>("axis", axis);
    return SingleGradientDef(
        "BatchGatherGradient",
        "",
        // This is the order as expected by BatchGatherGradient indices,
        // different from SpartseToDense above.
        vector<string>{I(Op::DATA), I(Op::INDICES), GO(0)},
        vector<string>{GI(0)},
        std::vector<Argument>{axisArg});
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Gather, GetGatherGradient);

} // namespace caffe2
