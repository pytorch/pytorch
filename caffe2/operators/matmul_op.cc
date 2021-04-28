#include "caffe2/operators/matmul_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MatMul, MatMulOp<float, CPUContext>);

OPERATOR_SCHEMA(MatMul)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      ArgumentHelper arg_helper(def);
      int axis_a = arg_helper.GetSingleArgument<int>("axis_a", 1);
      int axis_b = arg_helper.GetSingleArgument<int>("axis_b", 1);
      int trans_a = arg_helper.GetSingleArgument<bool>("trans_a", false);
      int trans_b = arg_helper.GetSingleArgument<bool>("trans_b", false);
      int canonical_axis_a = canonical_axis_index_(axis_a, in[0].dims().size());
      int canonical_axis_b = canonical_axis_index_(axis_b, in[0].dims().size());

      int M = size_to_dim_(canonical_axis_a, GetDimsVector(in[0]));
      int N = size_from_dim_(canonical_axis_b, GetDimsVector(in[1]));
      if (trans_a) {
        M = size_from_dim_(canonical_axis_a, GetDimsVector(in[0]));
      }
      if (trans_b) {
        N = size_to_dim_(canonical_axis_b, GetDimsVector(in[1]));
      }

      out[0].add_dims(M);
      out[0].add_dims(N);

      return out;
    })
    .SetDoc(R"DOC(
Matrix multiplication $Y = A * B$, where `A` has size (M x K), `B` has size
(K x N), and `Y` will have a size (M x N). To transpose `A` or `B` before
multiplication, pass 1 to the `trans_a` and/or `trans_b` arguments, which
separate the first and second dimensions of the respective matrices using
`axis_a` and `axis_b`.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/matmul_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "MatMul",
    ["A", "B"],
    ["Y"],
)

workspace.FeedBlob("A", np.random.randint(10, size=(3,3)).astype(np.float32))
workspace.FeedBlob("B", np.random.randint(10, size=(3,3)).astype(np.float32))
print("A:", workspace.FetchBlob("A"))
print("B:", workspace.FetchBlob("B"))
workspace.RunOperatorOnce(op)
print("Y:", workspace.FetchBlob("Y"))
```

**Result**

```
A: [[1. 8. 3.]
 [6. 4. 4.]
 [5. 4. 7.]]
B: [[4. 0. 3.]
 [3. 1. 1.]
 [8. 5. 8.]]
Y: [[52. 23. 35.]
 [68. 24. 54.]
 [88. 39. 75.]]
```

</details>

)DOC")
    .Input(
        0,
        "A",
        "*(type: Tensor`<float>`)* 2D matrix of size (M x K).")
    .Input(
        1,
        "B",
        "*(type: Tensor`<float>`)* 2D matrix of size (K x N).")
    .Output(
        0,
        "Y",
        "*(type: Tensor`<float>`)* 2D matrix of size (M x N).")
    .Arg(
        "axis_a",
        "*(type: int; default: 1)* Exclusive axis that divides the first and "
        "second dimension of matrix `A`.")
    .Arg(
        "axis_b",
        "*(type: int; default: 1)* Exclusive axis that divides the first and "
        "second dimension of matrix `B`.")
    .Arg(
        "trans_a",
        "*(type: int; default: 0)* Pass 1 to transpose `A` before multiplication and "
        "after the dimension adjustment using `axis_a`.")
    .Arg(
        "trans_b",
        "*(type: int; default: 0)* Pass 1 to transpose `B` before multiplication and "
        "after the dimension adjustment using `axis_b`.");

class GetMatMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(def_.input_size() == 2 || def_.input_size() == 3);

    bool axis_a = 1;
    bool axis_b = 1;
    bool trans_a = 0;
    bool trans_b = 0;

    if (ArgumentHelper::HasArgument(Def(), "trans_a")) {
      trans_a = GetArgument(Def(), "trans_a").i();
    }
    if (ArgumentHelper::HasArgument(Def(), "trans_b")) {
      trans_b = GetArgument(Def(), "trans_b").i();
    }
    if (ArgumentHelper::HasArgument(Def(), "axis_a")) {
      axis_a = GetArgument(Def(), "axis_a").i();
    }
    if (ArgumentHelper::HasArgument(Def(), "axis_b")) {
      axis_b = GetArgument(Def(), "axis_b").i();
    }

    if (trans_a) {
      if (trans_b) {
        // A'B':
        // dA = B'G', dB = G'A'
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(1), GO(0), I(0)},
                vector<string>{GI(0)},
                vector<Argument>{MakeArgument<int>("trans_a", 1),
                                 MakeArgument<int>("trans_b", 1),
                                 MakeArgument<int>("axis_a", axis_b)}),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(0), I(1)},
                vector<string>{GI(1)},
                vector<Argument>{MakeArgument<int>("trans_a", 1),
                                 MakeArgument<int>("trans_b", 1),
                                 MakeArgument<int>("axis_b", axis_a)})};
      } else {
        // A'B:
        // dA = BG', dB = AG
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(1), GO(0), I(0)},
                vector<string>{GI(0)},
                vector<Argument>{MakeArgument<int>("trans_b", 1),
                                 MakeArgument<int>("axis_a", axis_b)}),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(0), GO(0), I(1)},
                vector<string>{GI(1)},
                vector<Argument>{MakeArgument<int>("axis_a", axis_a)})};
      }
    } else {
      if (trans_b) {
        // AB':
        // dA = GB, dB = G'A
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(1), I(0)},
                vector<string>{GI(0)},
                vector<Argument>{MakeArgument<int>("axis_b", axis_b)}),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(0), I(1)},
                vector<string>{GI(1)},
                vector<Argument>{MakeArgument<int>("trans_a", 1),
                                 MakeArgument<int>("axis_b", axis_a)})};
      } else {
        // AB:
        // dA = GB', dB = A'G
        return vector<OperatorDef>{
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{GO(0), I(1), I(0)},
                vector<string>{GI(0)},
                vector<Argument>{MakeArgument<int>("trans_b", 1),
                                 MakeArgument<int>("axis_b", axis_b)}),
            CreateOperatorDef(
                "MatMul",
                "",
                vector<string>{I(0), GO(0), I(1)},
                vector<string>{GI(1)},
                vector<Argument>{MakeArgument<int>("trans_a", 1),
                                 MakeArgument<int>("axis_a", axis_a)})};
      }
    }
  }

  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(MatMul, GetMatMulGradient);

} // namespace caffe2
