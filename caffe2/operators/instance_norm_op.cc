#include "caffe2/operators/instance_norm_op.h"

#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

namespace {

template <typename T>
void ComputeFusedParams(
    const int64_t N,
    const int64_t C,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  ConstEigenArrayMap<T> mean_arr(mean, C, N);
  ConstEigenArrayMap<T> rstd_arr(rstd, C, N);
  ConstEigenVectorArrayMap<T> gamma_arr(gamma, C);
  ConstEigenVectorArrayMap<T> beta_arr(beta, C);
  EigenArrayMap<T> scale_arr(scale, C, N);
  EigenArrayMap<T> bias_arr(bias, C, N);
  scale_arr = rstd_arr.colwise() * gamma_arr;
  bias_arr = (-scale_arr * mean_arr).colwise() + beta_arr;
}

template <typename T>
void InstanceNormForwardNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y) {
  ConstEigenArrayMap<T> scale_arr(scale, C, N);
  ConstEigenArrayMap<T> bias_arr(bias, C, N);
  for (int64_t i = 0; i < N; ++i) {
    ConstEigenArrayMap<T> X_arr(X + i * HxW * C, C, HxW);
    EigenArrayMap<T> Y_arr(Y + i * HxW * C, C, HxW);
    Y_arr = (X_arr.colwise() * scale_arr.col(i)).colwise() + bias_arr.col(i);
  }
}

} // namespace

template <>
bool InstanceNormOp<float, CPUContext>::RunOnDeviceWithOrderNCHW(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    float* mean,
    float* rstd) {
  ConstEigenArrayMap<float> X_arr(X, HxW, N * C);
  for (int64_t i = 0; i < N * C; ++i) {
    const float mean_val = X_arr.col(i).mean();
    float rstd_val =
        std::max(X_arr.col(i).square().mean() - mean_val * mean_val, 0.0f);
    rstd_val = 1.0f / std::sqrt(rstd_val + epsilon_);
    const int64_t c = i % C;
    const float scale = gamma[c] * rstd_val;
    const float bias = beta[c] - scale * mean_val;
    for (int64_t j = 0; j < HxW; ++j) {
      Y[i * HxW + j] = scale * X[i * HxW + j] + bias;
    }
    mean[i] = mean_val;
    rstd[i] = rstd_val;
  }
  return true;
}

template <>
bool InstanceNormOp<float, CPUContext>::RunOnDeviceWithOrderNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    float* mean,
    float* rstd) {
  ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CPU));
  ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CPU));
  float* scale_data = scale_.template mutable_data<float>();
  float* bias_data = bias_.template mutable_data<float>();
  const float c = 1.0f / static_cast<float>(HxW);
  EigenArrayMap<float> mean_arr(mean, C, N);
  EigenArrayMap<float> rstd_arr(rstd, C, N);
  for (int64_t n = 0; n < N; ++n) {
    ConstEigenArrayMap<float> X_arr(X + n * HxW * C, C, HxW);
    mean_arr.col(n) = X_arr.col(0);
    rstd_arr.col(n) = X_arr.col(0).square();
    for (int64_t i = 1; i < HxW; ++i) {
      mean_arr.col(n) += X_arr.col(i);
      rstd_arr.col(n) += X_arr.col(i).square();
    }
  }
  mean_arr *= c;
  rstd_arr = ((rstd_arr * c - mean_arr.square()).max(0.0f) + epsilon_).rsqrt();
  ComputeFusedParams<float>(
      N, C, mean, rstd, gamma, beta, scale_data, bias_data);
  InstanceNormForwardNHWC<float>(N, C, HxW, X, scale_data, bias_data, Y);
  return true;
}

REGISTER_CPU_OPERATOR(InstanceNorm, InstanceNormOp<float, CPUContext>);

OPERATOR_SCHEMA(InstanceNorm)
    .NumInputs(3)
    .NumOutputs(1, 3)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
The *InstanceNorm* op applies Instance Normalization over a 4D input as described in [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).

$$output = \frac{input-\mu_{input}}{\sqrt{\sigma_{input}^2} + \epsilon}*scale + bias$$

Notice, two of the outputs are optional so there are three output cases for this op. Case 1: output; Case 2: output, saved_mean; Case 3: output, saved_mean, saved_inv_stdev.

Github Links:

- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.h
- https://github.com/caffe2/caffe2/blob/master/caffe2/operators/instance_norm_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "InstanceNorm",
    ["input", "scale", "bias"],
    ["output"],
    epsilon=1e-5,
)

workspace.FeedBlob("input", np.random.randn(2, 1, 3, 3).astype(np.float32))
print("input:\n", workspace.FetchBlob("input"), "\n")

workspace.FeedBlob("scale", np.array([1.5]).astype(np.float32))
print("scale: ", workspace.FetchBlob("scale"))

workspace.FeedBlob("bias", np.array([1.]).astype(np.float32))
print("bias: ", workspace.FetchBlob("bias"))

workspace.RunOperatorOnce(op)
print("output:\n", workspace.FetchBlob("output"))

```

**Result**

```

input:
 [[[[ 0.97856593 -1.1832817  -0.2540021 ]
   [-1.3315694  -0.7485018   0.3787225 ]
   [-0.6826597  -1.4637762   0.57116514]]]


 [[[-0.44948956  0.85544354 -0.9315333 ]
   [-0.37202677 -0.22266895 -0.27194235]
   [ 0.4948163  -0.7296504   1.3393803 ]]]]

scale:  [1.5]
bias:  [1.]
output:
 [[[[ 3.5017493  -0.3791256   1.2890853 ]
   [-0.6453266   0.40137637  2.4249308 ]
   [ 0.5195738  -0.8826599   2.7703972 ]]]


 [[[ 0.12639964  2.856744   -0.8821926 ]
   [ 0.28847694  0.60098207  0.49788612]
   [ 2.1021945  -0.45978796  3.869297  ]]]]

```

</details>

)DOC")
    .Arg(
        "epsilon",
        "*(type: float; default: 1e-5)* The epsilon value to use to avoid division by zero.")
    .Arg(
        "order",
        // NOLINTNEXTLINE(modernize-raw-string-literal)
        "*(type: string; default: \"NCHW\")* Specifies the order of the input data blob, where $N$ is batch size, $C$ is number of channels, $H$ is spatial height, and $W$ is spatial width. The only other valid option is \"NHWC\".")
    .Input(0, "input", "The input 4-dimensional NCHW tensor to be operated on.")
    .Input(1, "scale", "The input 1-dimensional scale tensor of size *C*.")
    .Input(2, "bias", "The input 1-dimensional bias tensor of size *C*.")
    .Output(
        0,
        "output",
        "The output 4-dimensional tensor of the same shape as input.")
    .Output(
        1,
        "saved_mean",
        "(Optional) Saved mean used during training to speed up gradient computation. Should not be used for testing.")
    .Output(
        2,
        "saved_inv_stdev",
        "(Optional) Saved inverse stdev used during training to speed up gradient computation. Should not be used for testing.");

} // namespace caffe2
