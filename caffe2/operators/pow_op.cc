#include "caffe2/operators/pow_op.h"

#include <cmath>
#include <string>
#include <vector>

namespace caffe2 {

namespace {

template <typename T>
void ComputePowGradient(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    const int* C_dims,
    const T* dC,
    const T* A,
    const T* B,
    const T* C,
    T* dA,
    T* dB,
    CPUContext* context) {
  const int A_size =
      std::accumulate(A_dims, A_dims + ndim, 1, std::multiplies<int>());
  const int B_size =
      std::accumulate(B_dims, B_dims + ndim, 1, std::multiplies<int>());
  const int C_size =
      std::accumulate(C_dims, C_dims + ndim, 1, std::multiplies<int>());
  math::Set<T, CPUContext>(A_size, T(0), dA, context);
  math::Set<T, CPUContext>(B_size, T(0), dB, context);
  std::vector<int> index(ndim, 0);
  for (int C_index = 0; C_index < C_size; ++C_index) {
    const int A_index =
        math::utils::GetIndexFromDims(ndim, A_dims, index.data());
    const int B_index =
        math::utils::GetIndexFromDims(ndim, B_dims, index.data());
    dA[A_index] += (B[B_index] == 0 ? 0 : C[C_index] / A[A_index]) *
        B[B_index] * dC[C_index];
    dB[B_index] += C[C_index] * std::log(A[A_index]) * dC[C_index];
    math::utils::IncreaseIndexInDims(ndim, C_dims, index.data());
  }
}

} // namespace

template <>
template <typename T>
bool PowGradientOp<TensorTypes<float>, CPUContext>::ComputeUnaryPowGradient(
    const int N,
    const T& exponent,
    const T* dY,
    const T* X,
    const T* Y,
    T* dX) {
  if (exponent == T(0)) {
    math::Set<T, CPUContext>(N, 0, dX, &context_);
  } else if (exponent == T(1)) {
    if (dX != dY) {
      context_.template Copy<T, CPUContext, CPUContext>(N, dY, dX);
    }
  } else if (exponent == T(2)) {
    if (X == nullptr) {
      EigenVectorMap<T>(dX, N) = ConstEigenVectorArrayMap<T>(Y, N).sqrt() *
          ConstEigenVectorArrayMap<T>(dY, N) * T(2);
    } else {
      EigenVectorMap<T>(dX, N) = ConstEigenVectorArrayMap<T>(X, N) *
          ConstEigenVectorArrayMap<T>(dY, N) * T(2);
    }
  } else if (exponent == T(-1)) {
    EigenVectorMap<T>(dX, N) = -ConstEigenVectorArrayMap<T>(Y, N).square() *
        ConstEigenVectorArrayMap<T>(dY, N);
  } else if (exponent == T(0.5)) {
    EigenVectorMap<T>(dX, N) = ConstEigenVectorArrayMap<T>(dY, N) /
        ConstEigenVectorArrayMap<T>(Y, N) * T(0.5);
  } else if (exponent == T(-0.5)) {
    EigenVectorMap<T>(dX, N) = ConstEigenVectorArrayMap<T>(Y, N).cube() *
        ConstEigenVectorArrayMap<T>(dY, N) * T(-0.5);
  } else {
    if (X == nullptr) {
      const T b = (exponent - T(1)) / exponent;
      EigenVectorMap<T>(dX, N) = ConstEigenVectorArrayMap<T>(Y, N).pow(b) *
          ConstEigenVectorArrayMap<T>(dY, N) * exponent;
    } else {
      EigenVectorMap<T>(dX, N) =
          ConstEigenVectorArrayMap<T>(X, N).pow(exponent - 1) *
          ConstEigenVectorArrayMap<T>(dY, N) * exponent;
    }
  }
  return true;
}

template <>
template <typename T>
bool PowGradientOp<TensorTypes<float>, CPUContext>::ComputeSinglePowBGradient(
    const int N,
    const T* dC,
    const T* A,
    const T* C,
    T* dB) {
  *dB = (ConstEigenVectorArrayMap<T>(C, N) *
         ConstEigenVectorArrayMap<T>(A, N).log() *
         ConstEigenVectorArrayMap<T>(dC, N))
            .sum();
  return true;
}

template <>
template <typename T>
bool PowGradientOp<TensorTypes<float>, CPUContext>::ComputeBinaryPowGradient(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const T* dC,
    const T* A,
    const T* B,
    const T* C,
    T* dA,
    T* dB) {
  if (A_dims == B_dims) {
    const int size = std::accumulate(
        A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
    ConstEigenVectorArrayMap<T> dC_arr(dC, size);
    ConstEigenVectorArrayMap<T> A_arr(A, size);
    ConstEigenVectorArrayMap<T> B_arr(B, size);
    ConstEigenVectorArrayMap<T> C_arr(C, size);
    EigenVectorMap<T>(dA, size) = A_arr.pow(B_arr - T(1)) * B_arr * dC_arr;
    EigenVectorMap<T>(dB, size) = C_arr * A_arr.log() * dC_arr;
    return true;
  }
  const int ndim = std::max(A_dims.size(), B_dims.size());
  std::vector<int> A_broadcast_dims(ndim);
  std::vector<int> B_broadcast_dims(ndim);
  std::vector<int> C_broadcast_dims(ndim);
  math::utils::ComputeBroadcastBinaryOpDims(
      A_dims.size(),
      A_dims.data(),
      B_dims.size(),
      B_dims.data(),
      A_broadcast_dims.data(),
      B_broadcast_dims.data(),
      C_broadcast_dims.data());
  ComputePowGradient<T>(
      ndim,
      A_broadcast_dims.data(),
      B_broadcast_dims.data(),
      C_broadcast_dims.data(),
      dC,
      A,
      B,
      C,
      dA,
      dB,
      &context_);
  return true;
}

REGISTER_CPU_OPERATOR(Pow, PowOp<TensorTypes<float>, CPUContext>);
REGISTER_CPU_OPERATOR(
    PowGradient,
    PowGradientOp<TensorTypes<float>, CPUContext>);

namespace {

std::vector<TensorShape> PowOpShapeInference(
    const OperatorDef& def,
    const std::vector<TensorShape>& in) {
  std::vector<TensorShape> out(1);
  out[0].set_data_type(in[0].data_type());
  ArgumentHelper helper(def);
  const bool broadcast = helper.GetSingleArgument<bool>("broadcast", false);
  if (helper.HasArgument("exponent") || broadcast) {
    out[0].mutable_dims()->CopyFrom(in[0].dims());
  } else {
    const std::vector<int> A_dims(in[0].dims().begin(), in[0].dims().end());
    const std::vector<int> B_dims(in[1].dims().begin(), in[1].dims().end());
    const std::vector<int> C_dims =
        elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            A_dims, B_dims);
    for (const int dim : C_dims) {
      out[0].add_dims(dim);
    }
  }
  return out;
}

} // namespace

OPERATOR_SCHEMA(Pow)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .TensorInferenceFunction(PowOpShapeInference)
    .SetDoc(R"DOC(
The *Pow* op takes an input data tensor $X$ and an exponent parameter *exponent*, which can be a scalar or another tensor. As output, it produces a single output data tensor $Y$, where the function $f(x) = x^{exponent}$ has been applied to $X$ elementwise.

Github Links:

- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.h
- https://github.com/pytorch/pytorch/blob/master/caffe2/operators/pow_op.cc


<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Pow",
    ["X", "exponent"],
    ["Y"],
)

workspace.FeedBlob("X", np.array([1,2,3,4,5,6]).astype(np.float32))
print("X: ", workspace.FetchBlob("X"))

workspace.FeedBlob("exponent", np.array([2]).astype(np.float32))
print("exponent: ", workspace.FetchBlob("exponent"))

workspace.RunOperatorOnce(op)
print("Y: ", workspace.FetchBlob("Y"))

```

**Result**

```

X:  [1. 2. 3. 4. 5. 6.]
exponent:  [2.]
Y:  [ 1.  4.  9. 16. 25. 36.]

```

</details>


)DOC")
    .Input(0, "X", "Input data blob to be operated on.")
    .Input(
        1,
        "exponent",
        "Exponent blob containing the exponent(s) for calculation. "
        "Do not use if setting exponent via argument.")
    .Output(0, "Y", "Output data blob with the same shape as the input.")
    .Arg(
        "exponent",
        "The exponent of the power function. Do not use if setting exponent "
        "via input.")
    .Arg("axis", "*(type: int; default: -1)*")
    .Arg("broadcast", "*(type: bool; default: False)*");

OPERATOR_SCHEMA(PowGradient)
    .NumInputs({2, 3, 4})
    .NumOutputs(1, 2)
    .AllowInplace({{0, 0}});

namespace {

class GetPowGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper arg_helper(def_);
    if (arg_helper.HasArgument("exponent")) {
      return SingleGradientDef(
          "PowGradient",
          "",
          I(0) == O(0) ? std::vector<std::string>{GO(0), O(0)}
                       : std::vector<std::string>{GO(0), I(0), O(0)},
          std::vector<std::string>{GI(0)});
    } else {
      return SingleGradientDef(
          "PowGradient",
          "",
          std::vector<std::string>{GO(0), I(0), I(1), O(0)},
          std::vector<std::string>{GI(0), GI(1)});
    }
  }
};

} // namespace

REGISTER_GRADIENT(Pow, GetPowGradient);

} // namespace caffe2
