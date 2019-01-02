#include "caffe2/operators/pool_op.h"

#include <limits>

#include "caffe2/operators/pool_op_util.h"
#include "caffe2/utils/cpu_neon.h"

namespace caffe2 {

#define CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_1D(T, kOrder)       \
  template <>                                                               \
  template <>                                                               \
  bool AveragePoolFunctor<CPUContext>::Forward<T, kOrder, 1>(               \
      const int N,                                                          \
      const int C,                                                          \
      const std::array<int, 1>& X_dims,                                     \
      const std::array<int, 1>& Y_dims,                                     \
      const std::array<int, 1>& kernel,                                     \
      const std::array<int, 1>& /* dilation */,                             \
      const std::array<int, 1>& stride,                                     \
      const std::array<int, 2>& pads,                                       \
      const T* X,                                                           \
      T* Y,                                                                 \
      CPUContext* /* context */) {                                          \
    if (count_include_pad) {                                                \
      pool_op_util::RunAveragePool1D<T, kOrder, true>(                      \
          N, C, X_dims[0], Y_dims[0], kernel[0], stride[0], pads[0], X, Y); \
    } else {                                                                \
      pool_op_util::RunAveragePool1D<T, kOrder, false>(                     \
          N, C, X_dims[0], Y_dims[0], kernel[0], stride[0], pads[0], X, Y); \
    }                                                                       \
    return true;                                                            \
  }
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_1D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_1D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_1D

template <>
template <>
bool AveragePoolFunctor<CPUContext>::Forward<float, StorageOrder::NCHW, 2>(
    const int N,
    const int C,
    const std::array<int, 2>& X_dims,
    const std::array<int, 2>& Y_dims,
    const std::array<int, 2>& kernel,
    const std::array<int, 2>& dilation,
    const std::array<int, 2>& stride,
    const std::array<int, 4>& pads,
    const float* X,
    float* Y,
    CPUContext* /* context */) {
  if (count_include_pad) {
    pool_op_util::RunAveragePool2D<float, StorageOrder::NCHW, true>(
        N,
        C,
        X_dims[0],
        X_dims[1],
        Y_dims[0],
        Y_dims[1],
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        pads[0],
        pads[1],
        X,
        Y);
  } else if (pool_op_util::IsNeon4x4p0s0Eligible(
                 X_dims[0],
                 X_dims[1],
                 Y_dims[0],
                 Y_dims[1],
                 kernel[0],
                 kernel[1],
                 stride[0],
                 stride[1],
                 pads[0],
                 pads[1],
                 pads[2],
                 pads[3],
                 dilation[0],
                 dilation[1],
                 X,
                 Y)) {
    pool_op_util::RunNeonAveragePool4x4p0s0NCHW(
        N, C, X_dims[0], X_dims[1], X, Y);
  } else {
    pool_op_util::RunAveragePool2D<float, StorageOrder::NCHW, false>(
        N,
        C,
        X_dims[0],
        X_dims[1],
        Y_dims[0],
        Y_dims[1],
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        pads[0],
        pads[1],
        X,
        Y);
  }
  return true;
}

template <>
template <>
bool AveragePoolFunctor<CPUContext>::Forward<float, StorageOrder::NHWC, 2>(
    const int N,
    const int C,
    const std::array<int, 2>& X_dims,
    const std::array<int, 2>& Y_dims,
    const std::array<int, 2>& kernel,
    const std::array<int, 2>& /* dilation */,
    const std::array<int, 2>& stride,
    const std::array<int, 4>& pads,
    const float* X,
    float* Y,
    CPUContext* /* context */) {
  if (count_include_pad) {
    pool_op_util::RunAveragePool2D<float, StorageOrder::NHWC, true>(
        N,
        C,
        X_dims[0],
        X_dims[1],
        Y_dims[0],
        Y_dims[1],
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        pads[0],
        pads[1],
        X,
        Y);
  } else {
    pool_op_util::RunAveragePool2D<float, StorageOrder::NHWC, false>(
        N,
        C,
        X_dims[0],
        X_dims[1],
        Y_dims[0],
        Y_dims[1],
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        pads[0],
        pads[1],
        X,
        Y);
  }
  return true;
}

#define CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_3D(T, kOrder) \
  template <>                                                         \
  template <>                                                         \
  bool AveragePoolFunctor<CPUContext>::Forward<T, kOrder, 3>(         \
      const int N,                                                    \
      const int C,                                                    \
      const std::array<int, 3>& X_dims,                               \
      const std::array<int, 3>& Y_dims,                               \
      const std::array<int, 3>& kernel,                               \
      const std::array<int, 3>& /* dilation */,                       \
      const std::array<int, 3>& stride,                               \
      const std::array<int, 6>& pads,                                 \
      const T* X,                                                     \
      T* Y,                                                           \
      CPUContext* /* context */) {                                    \
    if (count_include_pad) {                                          \
      pool_op_util::RunAveragePool3D<T, kOrder, true>(                \
          N,                                                          \
          C,                                                          \
          X_dims[0],                                                  \
          X_dims[1],                                                  \
          X_dims[2],                                                  \
          Y_dims[0],                                                  \
          Y_dims[1],                                                  \
          Y_dims[2],                                                  \
          kernel[0],                                                  \
          kernel[1],                                                  \
          kernel[2],                                                  \
          stride[0],                                                  \
          stride[1],                                                  \
          stride[2],                                                  \
          pads[0],                                                    \
          pads[1],                                                    \
          pads[2],                                                    \
          X,                                                          \
          Y);                                                         \
    } else {                                                          \
      pool_op_util::RunAveragePool3D<T, kOrder, false>(               \
          N,                                                          \
          C,                                                          \
          X_dims[0],                                                  \
          X_dims[1],                                                  \
          X_dims[2],                                                  \
          Y_dims[0],                                                  \
          Y_dims[1],                                                  \
          Y_dims[2],                                                  \
          kernel[0],                                                  \
          kernel[1],                                                  \
          kernel[2],                                                  \
          stride[0],                                                  \
          stride[1],                                                  \
          stride[2],                                                  \
          pads[0],                                                    \
          pads[1],                                                    \
          pads[2],                                                    \
          X,                                                          \
          Y);                                                         \
    }                                                                 \
    return true;                                                      \
  }
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_3D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_3D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_AVERAGE_POOL_FUNCTOR_FORWARD_3D

#define CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_1D(T, kOrder)         \
  template <>                                                             \
  template <>                                                             \
  bool MaxPoolFunctor<CPUContext>::Forward<T, kOrder, 1>(                 \
      const int N,                                                        \
      const int C,                                                        \
      const std::array<int, 1>& X_dims,                                   \
      const std::array<int, 1>& Y_dims,                                   \
      const std::array<int, 1>& kernel,                                   \
      const std::array<int, 1>& /* dilation */,                           \
      const std::array<int, 1>& stride,                                   \
      const std::array<int, 2>& pads,                                     \
      const T* X,                                                         \
      T* Y,                                                               \
      CPUContext* /* context */) {                                        \
    pool_op_util::RunMaxPool1D<T, kOrder>(                                \
        N, C, X_dims[0], Y_dims[0], kernel[0], stride[0], pads[0], X, Y); \
    return true;                                                          \
  }
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_1D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_1D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_1D

template <>
template <>
bool MaxPoolFunctor<CPUContext>::Forward<float, StorageOrder::NCHW, 2>(
    const int N,
    const int C,
    const std::array<int, 2>& X_dims,
    const std::array<int, 2>& Y_dims,
    const std::array<int, 2>& kernel,
    const std::array<int, 2>& dilation,
    const std::array<int, 2>& stride,
    const std::array<int, 4>& pads,
    const float* X,
    float* Y,
    CPUContext* /* context */) {
  if (pool_op_util::IsNeon2x2p0s0Eligible(
          X_dims[0],
          X_dims[1],
          Y_dims[0],
          Y_dims[1],
          kernel[0],
          kernel[1],
          stride[0],
          stride[1],
          pads[0],
          pads[1],
          pads[2],
          pads[3],
          dilation[0],
          dilation[1],
          X,
          Y)) {
    pool_op_util::RunNeonMaxPool2x2p0s0NCHW(N, C, X_dims[0], X_dims[1], X, Y);
  } else {
    pool_op_util::RunMaxPool2D<float, StorageOrder::NCHW>(
        N,
        C,
        X_dims[0],
        X_dims[1],
        Y_dims[0],
        Y_dims[1],
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        pads[0],
        pads[1],
        X,
        Y);
  }
  return true;
}

template <>
template <>
bool MaxPoolFunctor<CPUContext>::Forward<float, StorageOrder::NHWC, 2>(
    const int N,
    const int C,
    const std::array<int, 2>& X_dims,
    const std::array<int, 2>& Y_dims,
    const std::array<int, 2>& kernel,
    const std::array<int, 2>& /* dilation */,
    const std::array<int, 2>& stride,
    const std::array<int, 4>& pads,
    const float* X,
    float* Y,
    CPUContext* /* context */) {
  pool_op_util::RunMaxPool2D<float, StorageOrder::NHWC>(
      N,
      C,
      X_dims[0],
      X_dims[1],
      Y_dims[0],
      Y_dims[1],
      kernel[0],
      kernel[1],
      stride[0],
      stride[1],
      pads[0],
      pads[1],
      X,
      Y);
  return true;
}

#define CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_3D(T, kOrder) \
  template <>                                                     \
  template <>                                                     \
  bool MaxPoolFunctor<CPUContext>::Forward<T, kOrder, 3>(         \
      const int N,                                                \
      const int C,                                                \
      const std::array<int, 3>& X_dims,                           \
      const std::array<int, 3>& Y_dims,                           \
      const std::array<int, 3>& kernel,                           \
      const std::array<int, 3>& /* dilation */,                   \
      const std::array<int, 3>& stride,                           \
      const std::array<int, 6>& pads,                             \
      const T* X,                                                 \
      T* Y,                                                       \
      CPUContext* /* context */) {                                \
    pool_op_util::RunMaxPool3D<T, kOrder>(                        \
        N,                                                        \
        C,                                                        \
        X_dims[0],                                                \
        X_dims[1],                                                \
        X_dims[2],                                                \
        Y_dims[0],                                                \
        Y_dims[1],                                                \
        Y_dims[2],                                                \
        kernel[0],                                                \
        kernel[1],                                                \
        kernel[2],                                                \
        stride[0],                                                \
        stride[1],                                                \
        stride[2],                                                \
        pads[0],                                                  \
        pads[1],                                                  \
        pads[2],                                                  \
        X,                                                        \
        Y);                                                       \
    return true;                                                  \
  }
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_3D(float, StorageOrder::NCHW)
CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_3D(float, StorageOrder::NHWC)
#undef CAFFE2_SPECIALIZED_MAX_POOL_FUNCTOR_FORWARD_3D

constexpr char kAveragePoolDoc[] = R"DOC(
consumes an input blob and applies average pooling across the the blob according
to kernel sizes, stride sizes, pad lengths and dilation. Average pooling consists
of taking the average value of a subset of the input tensor according to the kernel
size and downsampling the data into the output blob for further processing. The
`brew` module has a wrapper for this operator for use in a `ModelHelper` object.

Pooling layers reduce the spatial dimensionality of the input blob. Each of the
output blob's dimensions will reduce according to:

$$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h


<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "AveragePool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
)

workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
```

**Result**

```
X:
 [[[[-0.2883434   0.43498734  0.05417408  1.912558    0.09390241
    -0.33173105]
   [ 1.633709    1.2047161   0.36964908  0.99961185  0.4184147
     0.9989975 ]
   [ 1.7644193   0.1789665   1.5812988  -0.6038542  -0.36090398
     0.33195344]
   [ 0.9457722  -0.95174325 -0.78124577  1.2062047   1.1903144
     0.2586746 ]
   [ 1.252104    0.32645547  1.8073524  -0.78397465  0.9978303
    -0.97614396]
   [ 0.5440196   1.5778259  -0.76750124  0.5051756   0.8838398
    -0.37085298]]]]

Y:
 [[[[0.7462672  0.83399826 0.2948959 ]
   [0.4843537  0.3506009  0.35500962]
   [0.9251013  0.19026303 0.13366827]]]]
```

</details>

)DOC";

constexpr char kMaxPoolDoc[] = R"DOC(
consumes an input blob and applies max pooling across the the blob according to
kernel sizes, stride sizes, pad lengths and dilation. Max pooling consists of
taking the maximum value of a subset of the input tensor according to the kernel
size and downsampling the data into the output blob for further processing. The
`brew` module has a wrapper for this operator for use in a `ModelHelper` object.

Pooling layers reduce the spatial dimensionality of the input blob. Each of the
output blob's dimensions will reduce according to:

$$dim_{out}=\frac{dim_{in}-kernel+2*pad}{stride}+1$$

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pool_op.cc
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/conv_pool_op_base.h

<details>

<summary> <b>Example</b> </summary>

**Code**

```
workspace.ResetWorkspace()

op = core.CreateOperator(
    "MaxPool",
    ["X"],
    ["Y"],
    kernel=2,
    stride=2,
)

workspace.FeedBlob("X", np.random.randn(1, 1, 6, 6).astype(np.float32)) // NCHW
print("X:\n", workspace.FetchBlob("X"), "\n")
workspace.RunOperatorOnce(op)
print("Y:\n", workspace.FetchBlob("Y"))
```

**Result**

```
X:
 [[[[-2.8534958e-01 -1.7719941e+00 -8.2277227e-04  1.1088650e+00
    -2.1476576e+00 -3.5070452e-01]
   [-9.0058845e-01 -3.0070004e-01 -1.7907504e+00 -7.1746534e-01
     1.2798511e+00 -3.2214901e-01]
   [ 1.5806322e+00  1.6845188e+00 -2.6633200e-01 -3.8576153e-01
    -9.6424848e-02 -3.9696163e-01]
   [ 1.2572408e-01  6.3612902e-01 -3.9554062e-01 -6.9735396e-01
    -9.1898698e-01 -1.9609968e-01]
   [-1.1587460e+00  2.4605224e+00 -1.5497679e+00  1.3020347e-01
    -8.1293899e-01 -7.8803545e-01]
   [ 1.4323474e+00  1.3618395e+00  9.8975077e-02 -1.1307785e-01
     7.2035044e-01  2.7642491e-01]]]]

Y:
 [[[[-0.28534958  1.108865    1.2798511 ]
   [ 1.6845188  -0.266332   -0.09642485]
   [ 2.4605224   0.13020347  0.72035044]]]]

```

</details>

)DOC";

std::function<void(OpSchema&)> AveragePoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = "AveragePool{dim} {pool_doc}";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{pool_doc}", kAveragePoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
    schema.Output(0, "Y", "*(type: Tensor`<float>`)* Output data tensor.");
    /*
    schema.Arg("kernel", "*(type: int)* Size of the window to take an average
    over."); schema.Arg("stride", "*(type: int)* Stride of the window.");
    schema.Arg("pad", "*(type: int)* Implicit zero padding to be added on both
    sides."); schema.Arg("dilation", "*(type: int)* Parameter that controls
    the stride of elements in the window."); schema.Arg("order", "*(type:
    string; default: 'NCHW')* Order of the blob dimensions.");
    */
  };
}

std::function<void(OpSchema&)> MaxPoolDocGenerator(const char* dim) {
  return [=](OpSchema& schema) {
    string doc = "MaxPool{dim} {pool_doc}";
    c10::ReplaceAll(doc, "{dim}", dim);
    c10::ReplaceAll(doc, "{pool_doc}", kMaxPoolDoc);
    schema.SetDoc(doc);
    schema.Input(
        0,
        "X",
        "*(type: Tensor`<float>`)* Input data tensor of shape NCHW or NHWC.");
    schema.Output(0, "Y", "*(type: Tensor`<float>`)* Output data tensor.");
    /*
    schema.Arg("kernel", "*(type: int)* Size of the window to take an average
    over."); schema.Arg("stride", "*(type: int)* Stride of the window.");
    schema.Arg("pad", "*(type: int)* Implicit zero padding to be added on both
    sides."); schema.Arg("dilation", "*(type: int)* Parameter that controls
    the stride of elements in the window."); schema.Arg("order", "*(type:
    string; default: 'NCHW')* Order of the blob dimensions.");
    */
  };
}
REGISTER_CPU_OPERATOR(
    AveragePool,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(AveragePool)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator(""))
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(
    AveragePool1D,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(AveragePool1D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("1D"))
    .InheritOnnxSchema("AveragePool");

REGISTER_CPU_OPERATOR(
    AveragePool2D,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(AveragePool2D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("2D"))
    .InheritOnnxSchema("AveragePool");

REGISTER_CPU_OPERATOR(
    AveragePool3D,
    PoolOp<float, CPUContext, AveragePoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(AveragePool3D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(AveragePoolDocGenerator("3D"))
    .InheritOnnxSchema("AveragePool");

REGISTER_CPU_OPERATOR(
    MaxPool,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(MaxPool)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator(""))
    .InheritOnnxSchema();

REGISTER_CPU_OPERATOR(
    MaxPool1D,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(MaxPool1D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("1D"))
    .InheritOnnxSchema("MaxPool");

REGISTER_CPU_OPERATOR(
    MaxPool2D,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(MaxPool2D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("2D"))
    .InheritOnnxSchema("MaxPool");

REGISTER_CPU_OPERATOR(
    MaxPool3D,
    PoolOp<float, CPUContext, MaxPoolFunctor<CPUContext>>);

OPERATOR_SCHEMA(MaxPool3D)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForPool)
    .FillUsing(MaxPoolDocGenerator("3D"))
    .InheritOnnxSchema("MaxPool");

} // namespace caffe2
