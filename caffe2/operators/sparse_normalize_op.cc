#include "caffe2/operators/sparse_normalize_op.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <>
bool SparseNormalizeOp<float, CPUContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, Input(INDICES));
}

template <>
template <typename SIndex>
bool SparseNormalizeOp<float, CPUContext>::DoRunWithType() {
  const auto* indices = Input(INDICES).template data<SIndex>();
  const auto* paramIn = Input(PARAM).template data<float>();
  auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<float>();
  const float kEps = 1e-12f;

  // n: number of sparse embeddings to be normalized
  auto n = Input(INDICES).numel();
  if (n == 0) {
    return true;
  }

  // embedding length, e.g. 32, 64, 128
  auto block_size = Input(PARAM).size_from_dim(1);
  for (int i = 0; i < n; ++i) {
    auto idx = indices[i];
    auto offsetIdx = idx * block_size;
    ConstEigenVectorMap<float> xVec(paramIn + offsetIdx, block_size);
    auto norm = xVec.template lpNorm<2>();

    if (use_max_norm_ && norm <= norm_) {
      continue;
    }

    math::Scale(
        block_size,
        norm_ / (norm + kEps),
        paramOut + offsetIdx,
        paramOut + offsetIdx,
        &context_);
  }
  return true;
}

REGISTER_CPU_OPERATOR(SparseNormalize, SparseNormalizeOp<float, CPUContext>);
OPERATOR_SCHEMA(SparseNormalize)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .Input(0, "param", "Parameters to be normalized")
    .Input(1, "indices", "Sparse indices")
    .Input(
        2,
        "grad",
        "Gradient computed (optional - not used, this argument is for backwards compatibility)")
    .Output(0, "output_param", "Normalized parameters")
    .EnforceOneToOneInplace()
    .Arg(
        "use_max_norm",
        "A bool variable to control whether to use max norm \
    or constant norm. When use_max_norm = false, constant norm is used so that \
    all the embedding vectors are scaled to have a L2 norm equals to A \
    (see blow argument norm=A). If use_max_norm = true, \
    max norm is used so that embedding is scaled so that its l2 norm is no larger \
    than A. If an embedding's norm is less than A originally, \
    the embedding is left unchanged.\
    The default is True.")
    .Arg("norm", "L2 norm of the embedding. The default is 1.0.")
    .SetDoc(R"DOC(
Given a sparse matrix, apply max_norm or constant_norm sparse regularization.
)DOC");

SHOULD_NOT_DO_GRADIENT(SparseNormalize);
} // namespace caffe2
