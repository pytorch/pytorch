#include <c10/util/accumulate.h>

#include "caffe2/operators/elementwise_mul_op.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace caffe2 {

namespace {

template <typename TGrad, typename TIn>
void ComputeMulGradient(
    const int ndim,
    const int* A_dims,
    const int* B_dims,
    const int* C_dims,
    const TGrad* dC,
    const TIn* A,
    const TIn* B,
    TGrad* dA,
    TGrad* dB,
    CPUContext* context) {
  const auto A_size = c10::multiply_integers(A_dims, A_dims + ndim);
  const auto B_size = c10::multiply_integers(B_dims, B_dims + ndim);
  const auto C_size = c10::multiply_integers(C_dims, C_dims + ndim);
  math::Set<TGrad, CPUContext>(A_size, TGrad(0), dA, context);
  math::Set<TGrad, CPUContext>(B_size, TGrad(0), dB, context);
  std::vector<int> index(ndim, 0);
  for (int C_index = 0; C_index < C_size; ++C_index) {
    const int A_index =
        math::utils::GetIndexFromDims(ndim, A_dims, index.data());
    const int B_index =
        math::utils::GetIndexFromDims(ndim, B_dims, index.data());
    dA[A_index] += dC[C_index] * B[B_index];
    dB[B_index] += dC[C_index] * A[A_index];
    math::utils::IncreaseIndexInDims(ndim, C_dims, index.data());
  }
}

// A : input not to broadcast whose size is common_size x broadcast_size
// B : input to broadcast whose size is common_size
void ComputeMulGradient(
    const int common_size,
    const int broadcast_size,
    const float* dC,
    const float* A,
    const float* B,
    float* dA,
    float* dB,
    CPUContext* context) {
  for (int i = 0; i < common_size; ++i) {
    caffe2::math::Scale(
        broadcast_size,
        B[i],
        dC + i * broadcast_size,
        dA + i * broadcast_size,
        context);
    caffe2::math::Dot(
        broadcast_size,
        dC + i * broadcast_size,
        A + i * broadcast_size,
        dB + i,
        context);
  }
}

void ComputeMulGradient(
    const int size,
    const float* dC,
    const float* A,
    const float* B,
    float* dA,
    float* dB) {
  for (int i = 0; i < size; ++i) {
    dA[i] = dC[i] * B[i];
    dB[i] = dC[i] * A[i];
  }
}

} // namespace

template <>
template <typename TGrad, typename TIn, typename TOut>
bool MulFunctor<CPUContext>::Backward(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const TGrad* dC,
    const TIn* A,
    const TIn* B,
    const TOut* /* C */,
    TGrad* dA,
    TGrad* dB,
    CPUContext* context) const {
  if (A_dims == B_dims) {
    const auto size = c10::multiply_integers(A_dims);
    math::Mul(size, dC, B, dA, context);
    math::Mul(size, dC, A, dB, context);
    return true;
  }

  const int ndim = std::max(A_dims.size(), B_dims.size());
  if (ndim == 0) {
    return true;
  }

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

  const int C_size = std::accumulate(
      C_broadcast_dims.cbegin(),
      C_broadcast_dims.cbegin() + ndim,
      1,
      // NOLINTNEXTLINE(modernize-use-transparent-functors)
      std::multiplies<int>());
  if (C_size == 0) {
    const auto A_size = c10::multiply_integers(A_dims);
    const auto B_size = c10::multiply_integers(B_dims);
    math::Set<TGrad, CPUContext>(A_size, TGrad(0), dA, context);
    math::Set<TGrad, CPUContext>(B_size, TGrad(0), dB, context);
    return true;
  }

  // Flatten dims as much as possible
  // We call A is broadcasted at dim d if A_broadcast_dims[d] <= 1
  // Two consecutive dims d and d+1 can be flattened if
  // A and B are broadcasted at dim d, or
  // A and B are broadcasted at dim d + 1, or
  // A is broadcasted at dim d and d + 1, or
  // B is broadcasted at dim d and d + 1, or
  // A and B are not broadcasted at dim d and d + 1
  std::vector<int> A_broadcast_dims_flattened, B_broadcast_dims_flattened,
      C_broadcast_dims_flattened;
  A_broadcast_dims_flattened.reserve(ndim);
  B_broadcast_dims_flattened.reserve(ndim);

  A_broadcast_dims_flattened.push_back(A_broadcast_dims[0]);
  B_broadcast_dims_flattened.push_back(B_broadcast_dims[0]);

  for (int i = 1; i < ndim; ++i) {
    int A_old = A_broadcast_dims_flattened.back();
    int B_old = B_broadcast_dims_flattened.back();
    int A_new = A_broadcast_dims[i];
    int B_new = B_broadcast_dims[i];
    if ((A_old == 1 && B_old == 1) || (A_new == 1 && B_new == 1) ||
        (A_old == 1 && A_new == 1) || (B_old == 1 && B_new == 1) ||
        (A_old > 1 && B_old > 1 && A_new > 1 && B_new > 1)) {
      A_broadcast_dims_flattened.back() *= A_new;
      B_broadcast_dims_flattened.back() *= B_new;
    } else {
      A_broadcast_dims_flattened.push_back(A_new);
      B_broadcast_dims_flattened.push_back(B_new);
    }
  }

  int ndim_flattened = A_broadcast_dims_flattened.size();
  C_broadcast_dims_flattened.resize(ndim_flattened);
  for (int i = 0; i < ndim_flattened; ++i) {
    C_broadcast_dims_flattened[i] =
        std::max(A_broadcast_dims_flattened[i], B_broadcast_dims_flattened[i]);
  }

  if (std::is_same<TGrad, float>::value && std::is_same<TIn, float>::value &&
      ndim_flattened <= 2 &&
      A_broadcast_dims_flattened[0] == B_broadcast_dims_flattened[0] &&
      (ndim_flattened == 1 || A_broadcast_dims_flattened[1] <= 1 ||
       B_broadcast_dims_flattened[1] <= 1)) {
    if (ndim_flattened == 2) {
      // fast path when we have 2 flattened dimensions and the second dimension
      // is broadcasted.
      bool broadcast_B = B_broadcast_dims_flattened[1] <= 1;
      ComputeMulGradient(
          C_broadcast_dims_flattened[0],
          C_broadcast_dims_flattened[1],
          reinterpret_cast<const float*>(dC),
          reinterpret_cast<const float*>(broadcast_B ? A : B),
          reinterpret_cast<const float*>(broadcast_B ? B : A),
          reinterpret_cast<float*>(broadcast_B ? dA : dB),
          reinterpret_cast<float*>(broadcast_B ? dB : dA),
          context);
    } else {
      // fast path when we have 1 flattened dimension
      assert(ndim_flattened == 1);
      ComputeMulGradient(
          C_broadcast_dims_flattened[0],
          reinterpret_cast<const float*>(dC),
          reinterpret_cast<const float*>(A),
          reinterpret_cast<const float*>(B),
          reinterpret_cast<float*>(dA),
          reinterpret_cast<float*>(dB));
    }
  } else {
    ComputeMulGradient<TGrad, TIn>(
        ndim_flattened,
        A_broadcast_dims_flattened.data(),
        B_broadcast_dims_flattened.data(),
        C_broadcast_dims_flattened.data(),
        dC,
        A,
        B,
        dA,
        dB,
        context);
  }

  return true;
}

// Used in fallback ops
template bool MulFunctor<CPUContext>::Backward<float, float, float>(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const float* dC,
    const float* A,
    const float* B,
    const float* /* C */,
    float* dA,
    float* dB,
    CPUContext* context) const;

template bool MulFunctor<CPUContext>::Backward<int32_t, int32_t, int32_t>(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const int* dC,
    const int* A,
    const int* B,
    const int* /* C */,
    int* dA,
    int* dB,
    CPUContext* context) const;

template bool MulFunctor<CPUContext>::Backward<double, double, double>(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const double* dC,
    const double* A,
    const double* B,
    const double* /* C */,
    double* dA,
    double* dB,
    CPUContext* context) const;

template bool MulFunctor<CPUContext>::Backward<int64_t, int64_t, int64_t>(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    const int64_t* dC,
    const int64_t* A,
    const int64_t* B,
    const int64_t* /* C */,
    int64_t* dA,
    int64_t* dB,
    CPUContext* context) const;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    MulGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        CPUContext,
        MulFunctor<CPUContext>>);

namespace {

class GetMulGradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "MulGradient",
        "",
        std::vector<std::string>{GO(0), I(0), I(1)},
        std::vector<std::string>{GI(0), GI(1)});
  }
};

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(Mul, GetMulGradient);

} // namespace caffe2
