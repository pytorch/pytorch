#include "ep_int8_fc_op.h"

#include <functional>

#include "caffe2/operators/fc_inference.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(EPInt8FC, EPInt8FCOp<CPUContext>);

using namespace std::placeholders;

OPERATOR_SCHEMA(EPInt8FC)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction([](auto && arg1, auto && arg2) { return FCShapeInference(arg1, arg2, false); })
    .CostInferenceFunction([](auto && arg1, auto && arg2) { return CostInferenceForFC(arg1, arg2, false); })
    .SetDoc(R"DOC(
The FC operator computes an output $(Y)$ as a linear combination of the input data blob $(X)$ with a weight blob $(W)$ and bias blob $(b)$. More formally,

$$Y = XW^T+b$$

Here, $X$ is a matrix of shape $(M,K)$, $W$ is a matrix of shape $(N,K)$, $b$ is a vector of length $N$, and $Y$ is a matrix of shape $(M,N)$. $N$ can be thought of as the number of nodes in the layer, $M$ is the batch size, and $K$ is the number of features in an input observation.

*NOTE: $X$ does not need to explicitly be a 2-dimensional matrix, however, if it is not it will be coerced into one. For an arbitrary $n$-dimensional tensor $X$, e.g. $[a_0, a_1, \ldots ,a_{k-1}, a_k, \ldots , a_{n-1}]$, where $a_i$ in $N$, and $k$ is the $axis$ arg provided, then $X$ will be coerced into a 2-dimensional tensor with dimensions $[a_0 * \ldots * a_{k-1}, a_k * \ldots * a_{n-1}]$. For the default case where axis=1, this means the $X$ tensor will be coerced into a 2D tensor of dimensions $[a_0, a_1 * \ldots * a_{n-1}]$, where $a_0$ is often the batch size. In this situation, we must have $a_0 = M$ and $a_1 * \ldots * a_{n-1} = K$. Lastly, even though $b$ is a vector of length $N$, it is copied and resized to shape $(M x N)$ implicitly, then added to each vector in the batch.*

<details>

<summary> <b>Example</b> </summary>

**Code**

```

// In this example, our batch size is 1 (M=1), the input observation will have
//   6 features (K=6), and the layer will have one hidden node (N=1). The
//   expected output is Y=7.
workspace.ResetWorkspace()

op = core.CreateOperator(
    "EPInt8FC",
    ["X", "W", "b"],
    ["Y"]
)

// Create X: MxK
data = np.array([1,2,3,4,5,6]).astype(np.float32)
data = data[np.newaxis,:]

// Create W: NxK
weights = np.array(np.array([1,1/2.,1/3.,1/4.,1/5.,1/6.])).astype(np.float32)
weights = weights[np.newaxis,:]

// Create b: N
bias = np.array([1.]).astype(np.float32)

// Put the inputs into the workspace
workspace.FeedBlob("X", data)
workspace.FeedBlob("W", weights)
workspace.FeedBlob("b", bias)

// Run the operator
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))

```

**Result**

```

Y:
 [[7.]]

```

</details>

)DOC")
    .Arg(
        "axis",
        "*(type: int; default: 1)* Describes the axis of the input data $X$. Defaults to one because in the common case when the input $X$ has shape $(M,K)$, the first axis encodes the batch size.")
    .Arg(
        "axis_w",
        "*(type: int; default: 1)* Describes the axis of the input weight matrix $W$. Defaults to one because the first axis most likely describes the batch_size.")
    .Arg(
        "float16_compute",
        "*(type: bool; default: False)* Whether to use float-16 compute kernel.")
    .Input(
        0,
        "X",
        "Input blob to be coerced into a 2D matrix of shape $(M,K)$, where $M$ is the batch size and $K$ is the number of features in a single observation.")
    .Input(
        1,
        "W",
        "Input blob to be coerced into a 2D matrix of shape $(N,K)$ describing a fully connected weight matrix. Here, $K$ is the number of features in a single observation and $N$ is the number of nodes in the FC layer.")
    .Input(
        2,
        "b",
        "Input blob containing vector of length $N$ which describes one bias for each node in the layer.")
    .Output(
        0,
        "Y",
        "Output blob containing a 2D output matrix of shape $(M,N)$, where $M$ is the batch size and $N$ is the number of nodes in the layer. The output is calculated as $Y=XW^T+b$.")
    .InheritOnnxSchema("Gemm");

} // namespace caffe2
