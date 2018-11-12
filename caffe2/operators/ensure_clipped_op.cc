#include "caffe2/operators/ensure_clipped_op.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename SIndex>
bool EnsureClippedOp<float, CPUContext>::DoRunWithType() {
  Output(OUTPUT_PARAM)->ResizeLike(Input(PARAM));
  const auto* indices = Input(INDICES).template data<SIndex>();
  const auto* paramIn = Input(PARAM).template data<float>();
  auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
  CAFFE_ENFORCE_EQ(paramIn, paramOut);
  // n: number of sparse embeddings to be normalized
  auto n = Input(INDICES).numel();
  if (n == 0) {
    return true;
  }
  // embedding length, e.g. 32, 64, 128
  auto block_size = Input(GRAD).numel() / n;
  for (int i = 0; i < n; ++i) {
    auto idx = indices[i];
    auto offsetIdx = idx * block_size;
    EigenVectorMap<float>(paramOut + offsetIdx, block_size) =
        ConstEigenVectorMap<float>(paramIn + offsetIdx, block_size)
            .cwiseMax(min_)
            .cwiseMin(max_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(EnsureClipped, EnsureClippedOp<float, CPUContext>);
OPERATOR_SCHEMA(EnsureClipped)
    .NumInputs(1, 3)
    .NumOutputs(1)
    .Input(0, "param", "Parameters to be normalized")
    .Input(1, "indices", "Sparse indices, only needed for sparse param")
    .Input(2, "grad", "Gradient computed, only needed for sparse param")
    .Output(0, "output_param", "param ensured to be clipped within range")
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Given a tensor, apply clip after gradient is applied; when the param is sparse as
indicated by valid indices and grad, in-place is required
)DOC");

SHOULD_NOT_DO_GRADIENT(EnsureClipped);
} // namespace caffe2
