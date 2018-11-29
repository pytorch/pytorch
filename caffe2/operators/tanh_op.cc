#include "caffe2/operators/tanh_op.h"

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

#ifdef CAFFE2_USE_ACCELERATE
template <>
template <>
bool TanhFunctor<CPUContext>::operator()<float>(
    const int N,
    const float* X,
    float* Y,
    CPUContext* /* context */) const {
  vvtanhf(Y, X, &N);
  return true;
}
#endif // CAFFE2_USE_ACCELERATE

REGISTER_CPU_OPERATOR(
    Tanh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        TanhFunctor<CPUContext>>);

namespace {
// Since the actual flops of the non-linear operator depends on the
// implementation, we use the number of non-linear operations as the proxy for
// the analytical flops for non-linear operator
OpSchema::Cost CostInferenceForTanh(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  CAFFE_ENFORCE_EQ(in.size(), 1, "Tanh requires one input");
  struct OpSchema::Cost c;
  ArgumentHelper helper(def);

  const uint64_t input_size =
      size_to_dim_(in[0].dims().size(), GetDimsVector(in[0]));

  const auto& X = in[0];
  c.flops = input_size;
  c.bytes_read = input_size * sizeof(X.data_type());
  c.bytes_written = input_size * sizeof(X.data_type());
  c.params_bytes = 0;
  return c;
}
} // namespace

using namespace std::placeholders;
OPERATOR_SCHEMA(Tanh)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .CostInferenceFunction(std::bind(CostInferenceForTanh, _1, _2))
    .SetDoc(R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tanh_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Tanh",
    ["X"],
    ["X"],
)

workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
print("X:\n", workspace.FetchBlob("X"), "\n")

workspace.RunOperatorOnce(op)
print("X:\n", workspace.FetchBlob("X"))

```

**Result**

```

X:
 [[ 2.032603   -2.3556721  -0.14955314]
 [ 0.39309832 -1.1020128  -0.92951244]
 [-0.62815386  0.21342885  1.4002231 ]]

X:
 [[ 0.9662601  -0.982175   -0.14844811]
 [ 0.3740282  -0.8012209  -0.73036647]
 [-0.55677974  0.21024609  0.8853999 ]]

```

</details>

)DOC")
    .Input(0, "input", "1-D input tensor")
    .Output(
        0,
        "output",
        "The hyperbolic tangent values of the input tensor, computed "
        "element-wise")
    .InheritOnnxSchema();

OPERATOR_SCHEMA(TanhGradient).NumInputs(2).NumOutputs(1).AllowInplace({{1, 0}});

} // namespace caffe2
