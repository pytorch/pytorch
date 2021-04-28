#include "caffe2/operators/space_batch_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SpaceToBatch, SpaceToBatchOp<CPUContext>);
OPERATOR_SCHEMA(SpaceToBatch).NumInputs(1).NumOutputs(1).SetDoc(R"DOC(
Zero-pads and then rearranges (permutes) blocks of spatial data into batch. More specifically, this op outputs a copy of the input tensor where values from the height and width dimensions are moved to the batch dimension. After the zero-padding is according to the `pad` argument, both height and width of the input must be divisible by the `block_size`. Only "NCHW" order is currently supported.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "SpaceToBatch",
    ["X"],
    ["Y"],
    pad=2,
    block_size=3
)

workspace.FeedBlob("X", np.random.rand(1,3,5,5).astype(np.float32))
print("X.shape:", workspace.FetchBlob("X").shape)
workspace.RunOperatorOnce(op)
print("Y.shape:", workspace.FetchBlob("Y").shape)

```

**Result**

```

X.shape: (1, 3, 5, 5)
Y.shape: (9, 3, 3, 3)

```

</details>

)DOC")
    .Arg("pad","(*int*): exclusive axis that divides the first and second dimension of matrix `A` (default=0)")
    .Arg("block_size","(*int*): height/width of spatial blocks to be moved (default=2)")
    .Arg("order","(*string*): order of dimensions of input and output blobs; only \"NCHW\" order is currently supported (default=\"NCHW\")")
    .Input(0,"X","(*Tensor`<float>`*): input tensor (NCHW order)")
    .Output(0,"Y","(*Tensor`<float>`*): output tensor (NCHW order)");

REGISTER_CPU_OPERATOR(BatchToSpace, BatchToSpaceOp<CPUContext>);
OPERATOR_SCHEMA(BatchToSpace).NumInputs(1).NumOutputs(1).SetDoc(R"DOC(
Rearranges (permutes) data from batch into blocks of spatial data, followed by cropping. This is the reverse transformation of `SpaceToBatch`. More specifically, this op outputs a copy of the input tensor where values from the batch dimension are moved in spatial blocks to the height and width dimensions, followed by cropping along the height and width dimensions. Only "NCHW" order is currently supported.

Github Links:
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/space_batch_op.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "BatchToSpace",
    ["X"],
    ["Y"],
    pad=3
)

workspace.FeedBlob("X", np.random.rand(10,3,32,32).astype(np.float32))
print("X.shape:", workspace.FetchBlob("X").shape)
workspace.RunOperatorOnce(op)
print("Y.shape:", workspace.FetchBlob("Y").shape)

```

**Result**

```

X.shape: (10, 3, 32, 32)
Y.shape: (2, 3, 58, 58)

```

</details>

)DOC")
    .Arg("pad","(*int*): exclusive axis that divides the first and second dimension of matrix `A` (default=0)")
    .Arg("block_size","(*int*): height/width of spatial blocks to be moved (default=2)")
    .Arg("order","(*string*): order of dimensions of input and output blobs; only \"NCHW\" order is currently supported (default=\"NCHW\")")
    .Input(0,"X","(*Tensor`<float>`*): input tensor (NCHW order)")
    .Output(0,"Y","(*Tensor`<float>`*): output tensor (NCHW order)");

class GetSpaceToBatchGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "BatchToSpace", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};

class GetBatchToSpaceGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SpaceToBatch", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(SpaceToBatch, GetSpaceToBatchGradient);
REGISTER_GRADIENT(BatchToSpace, GetBatchToSpaceGradient);
}
