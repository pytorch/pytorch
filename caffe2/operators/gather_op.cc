#include "gather_op.h"
namespace caffe2 {

REGISTER_CPU_OPERATOR(Gather, GatherOp<CPUContext>);

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
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      if (in[0].dims(0) == 0) {
        for (int i = 0; i < in[0].dims_size(); ++i) {
          out[0].add_dims(in[0].dims(i));
        }
      } else {
        for (auto d : in[1].dims()) {
          out[0].add_dims(d);
        }
        for (int i = 1; i < in[0].dims_size(); ++i) {
          out[0].add_dims(in[0].dims(i));
        }
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    });

class GetGatherGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    const bool dense_gradient =
        argsHelper.GetSingleArgument<bool>("dense_gradient", false);

    using Op = GatherOp<CPUContext>;

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
};
REGISTER_GRADIENT(Gather, GetGatherGradient);

} // namespace caffe2
