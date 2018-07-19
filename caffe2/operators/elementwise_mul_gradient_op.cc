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
  const int A_size =
      std::accumulate(A_dims, A_dims + ndim, 1, std::multiplies<int>());
  const int B_size =
      std::accumulate(B_dims, B_dims + ndim, 1, std::multiplies<int>());
  const int C_size =
      std::accumulate(C_dims, C_dims + ndim, 1, std::multiplies<int>());
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
    const int size = std::accumulate(
        A_dims.cbegin(), A_dims.cend(), 1, std::multiplies<int>());
    math::Mul(size, dC, B, dA, context);
    math::Mul(size, dC, A, dB, context);
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
  ComputeMulGradient<TGrad, TIn>(
      ndim,
      A_broadcast_dims.data(),
      B_broadcast_dims.data(),
      C_broadcast_dims.data(),
      dC,
      A,
      B,
      dA,
      dB,
      context);
  return true;
}

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

REGISTER_GRADIENT(Mul, GetMulGradient);

} // namespace caffe2
