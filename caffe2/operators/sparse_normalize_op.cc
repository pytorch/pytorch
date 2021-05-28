#include "caffe2/operators/sparse_normalize_op.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/eigen_utils.h"

#include "caffe2/utils/cpuid.h"

#ifdef USE_FBGEMM
#include "fbgemm/FbgemmConvert.h"
#endif

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

template <>
bool SparseNormalizeOp<c10::Half, CPUContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
      this, Input(INDICES));
}

inline void Float16ToFloat_ref(const at::Half* in, float* out, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    out[i] = in[i];
  }
}

template <>
template <typename SIndex>
bool SparseNormalizeOp<c10::Half, CPUContext>::DoRunWithType() {
  const auto* indices = Input(INDICES).template data<SIndex>();
  const auto* paramIn = Input(PARAM).template data<c10::Half>();
  auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<c10::Half>();
  const float kEps = 1e-12f;

  // n: number of sparse embeddings to be normalized
  auto n = Input(INDICES).numel();
  if (n == 0) {
    return true;
  }
  // embedding length, e.g. 32, 64, 128
  auto block_size = Input(PARAM).size_from_dim(1);
  vector<float> row_vec_fp32(block_size);
  auto out_data = row_vec_fp32.data();
  for (int i = 0; i < n; ++i) {
    auto idx = indices[i];
    auto offsetIdx = idx * block_size;
#ifdef USE_FBGEMM
    if (GetCpuId().avx2()) {
      fbgemm::Float16ToFloat_avx2(
          reinterpret_cast<const fbgemm::float16*>(paramIn + offsetIdx),
          out_data,
          block_size);
    } else {
      Float16ToFloat_ref(paramIn + offsetIdx, out_data, block_size);
    }
#else
    Float16ToFloat_ref(paramIn + offsetIdx, out_data, block_size);
#endif
    ConstEigenVectorMap<float> xVec_fp32(row_vec_fp32.data(), block_size);
    float norm = xVec_fp32.template lpNorm<2>();
    if (use_max_norm_ && norm <= norm_) {
      continue;
    }
    auto Y = paramOut + offsetIdx;
    EigenVectorArrayMap<c10::Half>(Y, block_size) *=
        static_cast<float>(norm_ / (norm + kEps));
  }
  return true;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(SparseNormalize, SparseNormalizeOp<float, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(SparseNormalize);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_CPU_OPERATOR(Float16SparseNormalize, SparseNormalizeOp<c10::Half, CPUContext>);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
OPERATOR_SCHEMA(Float16SparseNormalize)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SHOULD_NOT_DO_GRADIENT(Float16SparseNormalize);
} // namespace caffe2
