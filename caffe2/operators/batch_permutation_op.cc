#include "caffe2/operators/batch_permutation_op.h"

#include <cstring>
#include <vector>

#ifdef CAFFE2_USE_MKLDNN
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include <caffe2/ideep/utils/ideep_operator.h>
#endif

namespace caffe2 {

template <bool forwards>
void batch_permutation_loop(
    const int N,
    const int K,
    const float* src,
    const int* indices,
    float* dst) {
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  long numBytes = K * sizeof(float);
  if (forwards) {
#ifdef _OPENMP
#if (_OPENMP >= 201307)
#pragma omp parallel for simd
#else
#pragma omp parallel for
#endif
#endif
    for (int n = 0; n < N; n++) {
      int origIdx = n * K;
      int permuteIdx = indices[n] * K;
      std::memcpy(dst + origIdx, src + permuteIdx, numBytes);
    }
  } else {
    std::vector<int> backward_indices(N);
    for (int i = 0; i < N; ++i) {
      backward_indices[indices[i]] = i;
    }
    for (int n = 0; n < N; n++) {
      int permuteIdx = n * K;
      int origIdx = backward_indices[n] * K;
      std::memcpy(dst + permuteIdx, src + origIdx, numBytes);
    }
  }
}

template <>
bool BatchPermutationOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& indices = Input(1);

  CAFFE_ENFORCE(indices.dim() == 1, "indices must be 1-d");
  CAFFE_ENFORCE(
      X.dim32(0) == indices.dim32(0),
      "X.dim32(0) must be equal to indices.dim32(0)",
      "(",
      X.dim32(0),
      " vs. ",
      indices.dim32(0),
      ")");

  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  if (X.dim32(0) > 0) {
    batch_permutation_loop<true>(
        X.dim32(0),
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        X.numel() / X.dim32(0),
        X.data<float>(),
        indices.data<int>(),
        Y->mutable_data<float>());
  }
  return true;
}

template <>
bool BatchPermutationGradientOp<float, CPUContext>::RunOnDevice() {
  auto& indices = Input(0);
  auto& dY = Input(1);

  auto* dX = Output(0, dY.sizes(), at::dtype<float>());

  if (dY.dim32(0) > 0) {
    batch_permutation_loop<false>(
        dY.dim32(0),
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        dY.numel() / dY.dim32(0),
        dY.data<float>(),
        indices.data<int>(),
        dX->mutable_data<float>());
  }
  return true;
}

#ifdef CAFFE2_USE_MKLDNN
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(
    BatchPermutation,
    IDEEPFallbackOp<BatchPermutationOp<float, CPUContext>>);
#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(BatchPermutation, BatchPermutationOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(
    BatchPermutationGradient,
    BatchPermutationGradientOp<float, CPUContext>);

// Input: X, indices; Output: Y
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(BatchPermutation)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Batch permutation of an input tensor X given input indices. First dimension of
X equals batch size N. The indices stores a be permutation of N.
The output Y is a tensor of same shape as X, with data re-ordered according to
the indices within the batch size.

Example of batch permutation on a 2-D tensor with batch size 4:
  X = [
    [1, 5, 2, 3, 4, 6, 0],
    [4, 3, 3, 5, 2, 3, 1],
    [2, 2, 3, 6, 0, 0, 1],
    [0, 0, 1, 1, 2, 2, 3]
  ]
  indices = [2, 0, 1, 3]
  Y = [
    [2, 2, 3, 6, 0, 0, 1],
    [1, 5, 2, 3, 4, 6, 0],
    [4, 3, 3, 5, 2, 3, 1],
    [0, 0, 1, 1, 2, 2, 3]
  ]

Example of batch permutation on a 3-D tensor with batch size 4:
  X = [
    [[1, 5, 2], [3, 4, 6, 0]],
    [[4, 3, 3], [5, 2, 3, 1]],
    [[2, 2, 3], [6, 0, 0, 1]],
    [[0, 0, 1], [1, 2, 2, 3]]
  ]
  indices = [2, 0, 1, 3]
  Y = [
    [[2, 2, 3], [6, 0, 0, 1]],
    [[1, 5, 2], [3, 4, 6, 0]],
    [[4, 3, 3], [5, 2, 3, 1]],
    [[0, 0, 1], [1, 2, 2, 3]]
  ]
)DOC")
    .Input(0, "X", "Input tensor, where 1st dimension equals batch size")
    .Input(1, "indices", "Input indices of batch to permute")
    .Output(0, "Y", "Output permuted tensor");
// Input: indices, dY (aka "gradOutput"); Output: dX (aka "gradInput")
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(BatchPermutationGradient).NumInputs(2).NumOutputs(1);

class GetBatchPermutationGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "BatchPermutationGradient",
        "",
        vector<string>{I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_GRADIENT(BatchPermutation, GetBatchPermutationGradient);

} // namespace caffe2

using BatchPermutationOpFloatCPU =
    caffe2::BatchPermutationOp<float, caffe2::CPUContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    BatchPermutation,
    "_caffe2::BatchPermutation(Tensor X, Tensor indices) -> Tensor",
    BatchPermutationOpFloatCPU);
