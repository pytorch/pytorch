/*
  Provides the implementations of CUDA BLAS function templates.
 */

#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <c10/macros/Export.h>
#include <c10/util/irange.h>
#include <c10/core/ScalarType.h>

#include <ATen/cuda/detail/BLASConstants.h>
#include <ATen/cuda/detail/CublasLtUtils.h>

#ifdef USE_ROCM
#include <c10/cuda/CUDAStream.h>
#include <hipblaslt/hipblaslt-ext.hpp>
// until hipblas has an API to accept flags, we must use rocblas here
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#include <ATen/native/hip/ck_gemm.h>
#include <ATen/native/hip/ck_bgemm.h>
#define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 + ROCBLAS_VERSION_MINOR)
#define USE_GEMM_FLAGS_FP16_ALT_IMPL (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242)
// needed to work around calling rocblas API instead of hipblas API
static rocblas_operation hipOperationToRocOperation(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return rocblas_operation_none;
    case HIPBLAS_OP_T:
        return rocblas_operation_transpose;
    case HIPBLAS_OP_C:
        return rocblas_operation_conjugate_transpose;
    }
    TORCH_CHECK(false, "HIPBLAS_STATUS_INVALID_ENUM");
}
static hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    TORCH_CHECK(false, "HIPBLAS_STATUS_INVALID_ENUM");
}
// hipblas does not have hipblasSetMathMode
#define hipblasSetMathMode(handle, flags) HIPBLAS_STATUS_SUCCESS
// until we use hiblas v2
// hipify correctly maps things like CUDA_R_16F to HIP_R_16F,
// however hipblas v1 is still using its custom type
#ifndef HIPBLAS_V2
#define HIP_R_16F  HIPBLAS_R_16F
#define HIP_R_32F  HIPBLAS_R_32F
#define HIP_R_64F  HIPBLAS_R_64F
#define HIP_C_16F  HIPBLAS_C_16F
#define HIP_C_32F  HIPBLAS_C_32F
#define HIP_C_64F  HIPBLAS_C_64F
#define HIP_R_8I   HIPBLAS_R_8I
#define HIP_R_8U   HIPBLAS_R_8U
#define HIP_R_32I  HIPBLAS_R_32I
#define HIP_R_32U  HIPBLAS_R_32U
#define HIP_C_8I   HIPBLAS_C_8I
#define HIP_C_8U   HIPBLAS_C_8U
#define HIP_C_32I  HIPBLAS_C_32I
#define HIP_C_32U  HIPBLAS_C_32U
#define HIP_R_16BF HIPBLAS_R_16B
#define HIP_C_16BF HIPBLAS_C_16B
#endif
#endif

#define CUDABLAS_POSINT_CHECK(FD, X)         \
  TORCH_CHECK(                               \
      (X > 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X \
      " must be positive and less than ",    \
      INT_MAX,                               \
      " but got ",                           \
      X)

#define CUDABLAS_NONNEGINT_CHECK(FD, X)       \
  TORCH_CHECK(                                \
      (X >= 0 && X <= INT_MAX),               \
      "at::cuda::blas::" #FD " argument " #X  \
      " must be non-negative and less than ", \
      INT_MAX,                                \
      " but got ",                            \
      X)

namespace {

void _cublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).

  // Q: Why does Level3 check trans but this doesn't?
  // A: In level 2, the sizes (m, n) specify the size of A
  // (independent of trans value). In level 3. the sizes (m, n, k)
  // specify the sizes of op(A), op(B) where op depend on trans
  // values.
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12000
class CuBlasSMCarveoutGuard {
 public:
  explicit CuBlasSMCarveoutGuard(cublasHandle_t handle) : handle_(handle) {
    const auto smCarveout = at::globalContext()._SMCarveout_EXPERIMENTAL();
    if (smCarveout.has_value()) {
      const int smCount =
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
      const int carveout = smCarveout.value();
      TORCH_CHECK(
          carveout >= 0 && carveout < smCount,
          "SM carveout must be between 0 and ",
          smCount - 1,
          " (num_sms-1), got ",
          carveout);
      TORCH_CUDABLAS_CHECK(
          cublasSetSmCountTarget(handle_, smCount - carveout));
      enabled_ = true;
    }
  }
  ~CuBlasSMCarveoutGuard() {
    if (enabled_) {
      cublasSetSmCountTarget(handle_, 0);
    }
  }
  CuBlasSMCarveoutGuard(const CuBlasSMCarveoutGuard&) = delete;
  CuBlasSMCarveoutGuard& operator=(const CuBlasSMCarveoutGuard&) = delete;

 private:
  cublasHandle_t handle_;
  bool enabled_ = false;
};
#define CUBLAS_SM_CARVEOUT_GUARD(handle) CuBlasSMCarveoutGuard smCarveoutGuard(handle)
#else
#define CUBLAS_SM_CARVEOUT_GUARD(handle)
#endif

#ifdef USE_ROCM
static c10::cuda::CUDAStream _getCarveoutStream(int32_t value) {
  // 0 is default value, meaning full CUs i.e. no mask
  if (value == 0) {
    return at::cuda::getCurrentCUDAStream();
  }
  static int32_t last_value = 0;
  static hipStream_t stream;
  if (last_value == 0) {
    // first request, do nothing for this case
  }
  else if (last_value == value) {
    // stream was created previously and value hasn't changed
    return c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
  }
  else {
    // need a new stream and a previous stream exists, delete it
    AT_CUDA_CHECK(hipStreamDestroy(stream));
  }

  // if we got here, we need to create a new stream
  int32_t CUs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // how many uint32_t do we need to cover all CUs, fill bitmask with 1
  uint32_t mask_size = static_cast<uint32_t>((CUs + 32 - 1) / 32);
  std::vector<uint32_t> mask(mask_size, uint32_t{0x00000000});
  // starting from lowest order bits, in 32-bit chunks
  // set bits to 0 based on how many CUs to carve out
  int32_t full_shifts = value / 32;
  int32_t remainder = value % 32;
  for (int32_t i = 0; i < full_shifts; i++) {
    mask[i] = uint32_t{0xffffffff};
  }
  mask[full_shifts] = uint32_t{0xffffffff} << (32 - remainder);

  // finally, create masked stream
  AT_CUDA_CHECK(hipExtStreamCreateWithCUMask(&stream, mask_size, &mask[0]));

  last_value = value;
  return c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
}

static void _syncCurrentWithCarveoutStream(hipStream_t stream, bool presync) {
  hipEvent_t event;
  AT_CUDA_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));

  auto current_stream = at::cuda::getCurrentCUDAStream();

  if (presync) {
    AT_CUDA_CHECK(hipEventRecord(event, current_stream));
    AT_CUDA_CHECK(hipStreamWaitEvent(stream, event, 0));
  }
  else {
    AT_CUDA_CHECK(hipEventRecord(event, stream));
    AT_CUDA_CHECK(hipStreamWaitEvent(current_stream, event, 0));
  }
}
#endif

} // anonymous namespace

namespace at::cuda::blas {

using detail::CuBlasLtMatmulDescriptor;
using detail::CuBlasLtMatrixLayout;
using detail::CuBlasLtMatmulPreference;
using detail::CublasLtWorkspace;
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
using detail::CuBlasLtGroupedMatrixLayout;
#endif

/* LEVEL 3 BLAS FUNCTIONS */

#define GEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
  } while (0)

#define BGEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldc);  \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, num_batches);  \
  } while (0)

template <typename Dtype, typename C_Dtype = Dtype>
static inline bool bgemm_internal_cublaslt(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
#if defined(USE_ROCM) && ROCM_VERSION == 60400
  // regression in ROCm 6.4, planned fixed in 6.4.1, hipblaslt TT fp32 calculation errors
  // best to disallow hipblaslt for this specific case
  if constexpr (std::is_same_v<Dtype, float>) {
    if (detail::cublasOpFromChar(transa) == CUBLAS_OP_T && detail::cublasOpFromChar(transb) == CUBLAS_OP_T) {
        return false;
    }
  }
#endif
  const auto type_info = detail::getCublasLtTypeInfo<Dtype, C_Dtype>();
  const cudaDataType_t abType = type_info.ab_type;
  const cudaDataType_t cType = type_info.c_type;
  const cublasComputeType_t computeType = type_info.compute_type;
  const cudaDataType_t scaleType = type_info.scale_type;
  CuBlasLtMatmulPreference preference;
#ifndef USE_ROCM
  at::Half halpha;
  at::Half hbeta;
  uint32_t mask = -1;
#endif
  void * alpha_ptr = &alpha;
  void * beta_ptr = &beta;
  if constexpr (std::is_same_v<Dtype, at::Half>) {
#ifndef USE_ROCM
    if (computeType == CUBLAS_COMPUTE_16F) {
      halpha = alpha;
      hbeta = beta;
      alpha_ptr = &halpha;
      beta_ptr = &hbeta;
    }
#endif
  }
#ifndef USE_ROCM
  if constexpr (std::is_same_v<Dtype, at::Half>) {
    auto fp16_reduction = at::globalContext().allowFP16ReductionCuBLAS();
    if (fp16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      mask =
          fp16_reduction ==
                  at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
              ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                 CUBLASLT_REDUCTION_SCHEME_NONE)
              : CUBLASLT_REDUCTION_SCHEME_NONE;
      preference.setAttribute(
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
    }
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    auto bf16_reduction = at::globalContext().allowBF16ReductionCuBLAS();
    if (bf16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      mask =
          bf16_reduction ==
                  at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
              ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                 CUBLASLT_REDUCTION_SCHEME_NONE)
              : CUBLASLT_REDUCTION_SCHEME_NONE;
      preference.setAttribute(
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
    }
  }
#endif

  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);
  auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    computeDesc.setAttribute<int32_t>(
        CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
            at::globalContext()._SMCarveout_EXPERIMENTAL().value());
  }
#else
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    stream = _getCarveoutStream(
        at::globalContext()._SMCarveout_EXPERIMENTAL().value());
    _syncCurrentWithCarveoutStream(stream, true);
  }
#endif
  CuBlasLtMatrixLayout Adesc(abType, m, k, lda, opa != CUBLAS_OP_N);
  CuBlasLtMatrixLayout Bdesc(abType, k, n, ldb, opb != CUBLAS_OP_N);
  CuBlasLtMatrixLayout Cdesc(cType, m, n, ldc);

  if (num_batches > 1) {
    int num_batches_as_int = static_cast<int>(num_batches);
    Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridea);
    Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideb);
    Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridec);
  }

#ifndef USE_ROCM
  uint32_t a_alignment = detail::getAlignment(reinterpret_cast<uintptr_t>(a));
  uint32_t b_alignment = detail::getAlignment(reinterpret_cast<uintptr_t>(b));
  uint32_t c_alignment = detail::getAlignment(reinterpret_cast<uintptr_t>(c));
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
#endif

  auto ltworkspace = CublasLtWorkspace();
  TORCH_CHECK(ltworkspace.ptr != nullptr, "OOM trying to allocate workspace for cublaslt");
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ltworkspace.size);

  cublasStatus_t cublasStatus = CUBLAS_STATUS_SUCCESS;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  // on Blackwell+, we fake a n > 1 matmul when querying heuristics
  // to prevent cuBLASLt from dispatching to a GEMV kernel for batch-invariance
#ifndef USE_ROCM
  const bool lie_to_cublaslt = mask == CUBLASLT_REDUCTION_SCHEME_NONE && n == 1 && at::cuda::getCurrentDeviceProperties()->major >= 10;
#else
  const bool lie_to_cublaslt = false;
#endif
  if (lie_to_cublaslt) {
     const auto fake_ldb = ldb == 1 ? 2 : ldb;
     const auto fake_ldc = ldc == 1 ? 2 : ldc;
     CuBlasLtMatrixLayout FakeBdesc(abType, k, 2, fake_ldb, opb == CUBLAS_OP_T);
     CuBlasLtMatrixLayout FakeCdesc(cType, m, 2, fake_ldc);
     if (num_batches > 1) {
       int num_batches_as_int = static_cast<int>(num_batches);
       FakeBdesc.setAttribute(
           CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
       FakeCdesc.setAttribute(
           CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
       FakeBdesc.setAttribute(
           CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideb);
       FakeCdesc.setAttribute(
           CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridec);
     }

     TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        computeDesc.descriptor(),
        Adesc.descriptor(),
        FakeBdesc.descriptor(),
        FakeCdesc.descriptor(),
        FakeCdesc.descriptor(),
        preference.descriptor(),
        1,
        &heuristicResult,
        &returnedResult));
  } else {
    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        computeDesc.descriptor(),
        Adesc.descriptor(),
        Bdesc.descriptor(),
        Cdesc.descriptor(),
        Cdesc.descriptor(),
        preference.descriptor(),
        1,
        &heuristicResult,
        &returnedResult));
  }
  if (returnedResult == 0) {
    cublasStatus = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  else {
    cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      alpha_ptr,
      a,
      Adesc.descriptor(),
      b,
      Bdesc.descriptor(),
      beta_ptr,
      c,
      Cdesc.descriptor(),
      c,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      ltworkspace.ptr,
      ltworkspace.size,
      stream);
#ifdef USE_ROCM
    if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
      _syncCurrentWithCarveoutStream(stream, false);
    }
#endif
  }
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    TORCH_WARN(
      "bgemm_internal_cublaslt error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transa,
      " transpose_mat2 ",
      transb,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " lda ",
      lda,
      " ldb ",
      ldb,
      " ldc ",
      ldc,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType,
      ". Will attempt to recover by calling cublas instead.");
    return false;
  }
  return true;
}


template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_internal_cublas(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::bgemm: not implemented for input type ", typeid(Dtype).name(), " and output type ", typeid(C_Dtype).name());
}

template <>
void bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(cublasDgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, stridea, reinterpret_cast<const cuDoubleComplex*>(b), ldb, strideb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(cublasCgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, stridea, reinterpret_cast<const cuComplex*>(b), ldb, strideb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc, stridec, num_batches));
}

template <typename C_Dtype>
inline void bgemm_internal_cublas_half_helper(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, C_Dtype)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(at::Half);
  float falpha = alpha;
  float fbeta = beta;
#ifndef USE_ROCM
  at::Half halpha;
  at::Half hbeta;
  auto compute_type = CUDA_R_32F;
#endif
  void * alpha_ptr = &falpha;
  void * beta_ptr = &fbeta;
#ifdef USE_ROCM
  int flag = 0;
  rocblas_datatype c_type = std::is_same<C_Dtype, float>::value ? rocblas_datatype_f32_r : rocblas_datatype_f16_r;
  rocblas_datatype d_type = c_type;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                   hipOperationToRocOperation(opa),
                                   hipOperationToRocOperation(opb), (int)m, (int)n, (int)k,
                                   (void*)alpha_ptr, a, rocblas_datatype_f16_r, (int)lda, stridea,
                                   b, rocblas_datatype_f16_r, (int)ldb, strideb,
                                   (void*)beta_ptr, c, c_type, (int)ldc, stridec,
                                   c, d_type, (int)ldc, stridec,
                                   (int) num_batches, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                   0, flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 7 && at::globalContext().allowFP16AccumulationCuBLAS()) {
    halpha = alpha;
    hbeta = beta;
    compute_type = CUDA_R_16F;
    alpha_ptr = &halpha;
    beta_ptr = &hbeta;
  }
  if (prop->major >= 5){
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, opa, opb, m, n, k,
      alpha_ptr, a, CUDA_R_16F, lda, stridea,
      b, CUDA_R_16F, ldb, strideb, beta_ptr,
      c, std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16F, ldc, stridec,
      num_batches, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    for (const auto i : c10::irange(num_batches)) {
      if (std::is_same_v<C_Dtype, float>) {
        float* c_ptr = (float*)(c + i * stridec);
        at::cuda::blas::gemm<at::Half, float>(
            transa, transb,
            m, n, k,
            alpha, (a + i * stridea), lda,
            (b + i * strideb), ldb, beta,
            c_ptr, ldc);
      } else {
        at::cuda::blas::gemm<at::Half>(
            transa, transb,
            m, n, k,
            alpha, (a + i * stridea), lda,
            (b + i * strideb), ldb, beta,
            (c + i * stridec), ldc);
      }
    }
  }
#endif // USE_ROCM
}

template <typename C_Dtype>
inline void bgemm_internal_cublas_bfloat16_helper(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, C_Dtype)) {
  BGEMM_CHECK_ARGVALUES(at::BFloat16);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  const float falpha = alpha;
  const float fbeta = beta;
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

#if defined(USE_ROCM)
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                              opa, opb, (int)m, (int)n, (int)k,
                              (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea,
                              b, CUDA_R_16BF, (int)ldb, strideb,
                              (void*)&fbeta, c, std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16BF,
                              (int)ldc, stridec, (int)num_batches,
                              compute_type,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <>
void bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  bgemm_internal_cublas_half_helper<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
}

template <>
void bgemm_internal_cublas<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  bgemm_internal_cublas_half_helper<float>(CUDABLAS_BGEMM_ARGS(at::Half));
}

template <>
void bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  bgemm_internal_cublas_bfloat16_helper<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}


template <>
void bgemm_internal_cublas<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  bgemm_internal_cublas_bfloat16_helper<float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}


template <>
void bgemm_internal<double>(CUDABLAS_BGEMM_ARGTYPES(double))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support double gemm yet
    bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
#else
    if (!bgemm_internal_cublaslt<double>(CUDABLAS_BGEMM_ARGS(double))) {
      bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
    }
#endif
  }
  else {
    bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
  }
}

template <>
void bgemm_internal<float>(CUDABLAS_BGEMM_ARGTYPES(float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<float>(CUDABLAS_BGEMM_ARGS(float))) {
      bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGS(float));
    }
  }
  else {
    bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}

template <>
void bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex<double> gemm yet
    bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
#else
    if (!bgemm_internal_cublaslt<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>))) {
      bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
    }
#endif
  }
  else {
    bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
}

template <>
void bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex<float> gemm yet
    bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
#else
    if (!bgemm_internal_cublaslt<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>))) {
      bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
    }
#endif
  }
  else {
    bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
}

template <>
void bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half))
{
  auto preferred = at::globalContext().blasPreferredBackend();
#ifdef USE_ROCM
  // hipBLASLt has no equivalent of rocblas_gemm_flags_fp16_alt_impl on gfx90a,
  // so a backward fp16 GEMM on the hipBLASLt path silently flushes subnormals
  // (issue #182952). Route fp16 backward GEMMs to rocBLAS so the
  // ROCmBackwardPassGuard / fp16_alt_impl handling still applies.
  if (preferred == BlasBackend::Cublaslt &&
      at::ROCmBackwardPassGuard::is_backward_pass() &&
      at::detail::getCUDAHooks().isGPUArch({"gfx90a"})) {
    preferred = BlasBackend::Cublas;
  }
#endif
  if (preferred == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half))) {
      bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
    }
  }
  else {
    bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16))) {
      bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    }
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::bgemm_internal_ck<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
#endif
  else {
    bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template<>
void bgemm_internal<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float))
{
  if (at::globalContext().allowFP16AccumulationCuBLAS()) {
    // Do not allow fp16 reductions with fp32 output
    TORCH_CHECK(false, "bgemm input type at::Half and output type float is not supported with allowFP16AccumulationCuBLAS");
  }

  auto preferred = at::globalContext().blasPreferredBackend();
#ifdef USE_ROCM
  // See bgemm_internal<at::Half>: reroute fp16 backward GEMMs off hipBLASLt on
  // gfx90a to preserve subnormals (issue #182952).
  if (preferred == BlasBackend::Cublaslt &&
      at::ROCmBackwardPassGuard::is_backward_pass() &&
      at::detail::getCUDAHooks().isGPUArch({"gfx90a"})) {
    preferred = BlasBackend::Cublas;
  }
#endif
  if (preferred == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half))) {
      bgemm_internal_cublas<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half));
    }
  }
#if defined(USE_ROCM) && !defined(_MSC_VER)
  else if (preferred == BlasBackend::Ck) {
    TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported for ROCm");
  }
#endif
  else {
    bgemm_internal_cublas<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template<>
void bgemm_internal<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    if (!bgemm_internal_cublaslt<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16))) {
      bgemm_internal_cublas<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
    }
  }
#if defined(USE_ROCM) && !defined(_MSC_VER)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    TORCH_CHECK(false, "gemm input type at::BFloat16 and output type float is not supported for ROCm");
  }
#endif
  else {
    bgemm_internal_cublas<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template <typename Dtype, typename C_Dtype = Dtype>
inline void bgemm_tunable(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  tunable::GemmStridedBatchedParams<Dtype> params;
  params.transa = transa;
  params.transb = transb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = alpha;
  params.a = a;
  params.lda = lda;
  params.stride_a = stridea;
  params.b = b;
  params.ldb = ldb;
  params.stride_b = strideb;
  params.beta = beta;
  params.c = c;
  params.ldc = ldc;
  params.stride_c = stridec;
  params.batch = num_batches;

  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  if (transa_ && transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::T, tunable::BlasOp::T> bgemm{};
    bgemm(&params);
  }
  else if (transa_ && !transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::T, tunable::BlasOp::N> bgemm{};
    bgemm(&params);
  }
  else if (!transa_ && transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::N, tunable::BlasOp::T> bgemm{};
    bgemm(&params);
  }
  else if (!transa_ && !transb_) {
    static tunable::GemmStridedBatchedTunableOp<Dtype, tunable::BlasOp::N, tunable::BlasOp::N> bgemm{};
    bgemm(&params);
  }
  else {
    TORCH_CHECK(false, "unreachable");
  }
}

template <>
void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<double>(CUDABLAS_BGEMM_ARGS(double));
  }
  else {
    bgemm_internal<double>(CUDABLAS_BGEMM_ARGS(double));
  }
}

template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<float>(CUDABLAS_BGEMM_ARGS(float));
  }
  else {
    bgemm_internal<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}

template <>
void bgemm<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
  else {
    bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
}

template <>
void bgemm<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
  else {
    bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
}

template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
  else {
    bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    bgemm_tunable<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
  else {
    bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template <>
void bgemm<at::Half, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  // TODO: Support tuning for Half inputs and FP32 output
  bgemm_internal<at::Half, float>(CUDABLAS_BGEMM_ARGS(at::Half));
}


template <>
void bgemm<at::BFloat16, float>(CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  #ifndef USE_ROCM
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();

    if (prop->major < 8)
      TORCH_CHECK(false, "bgemm input type at::BFloat16 and output type float is only supported for CUDA devices with compute capability 8.0 or higher");
  #endif
  // TODO: Support tuning for BFloat16 inputs and FP32 output
  bgemm_internal<at::BFloat16, float>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}



template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal_cublas(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  TORCH_CHECK(false, "at::cuda::blas::gemm: not implemented for input type ", typeid(Dtype).name(), " and output type ", typeid(C_Dtype).name());
}

template <>
void gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(cublasDgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(cublasZgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc));
}

template <>
void gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(cublasCgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc));
}

template <typename C_Dtype>
inline void gemm_internal_cublas_half_helper(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, C_Dtype)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
#ifndef USE_ROCM
  at::Half halpha;
  at::Half hbeta;
  auto compute_type = CUDA_R_32F;
#endif
  void * alpha_ptr = &falpha;
  void * beta_ptr = &fbeta;
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::Half);
#ifdef USE_ROCM
  int flag = 0;
  rocblas_datatype c_type = std::is_same<C_Dtype, float>::value ? rocblas_datatype_f32_r : rocblas_datatype_f16_r;
  rocblas_datatype d_type = c_type;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_ex(
      (rocblas_handle)handle,
      hipOperationToRocOperation(opa),
      hipOperationToRocOperation(opb),
      m,
      n,
      k,
      alpha_ptr,
      a,
      rocblas_datatype_f16_r,
      lda,
      b,
      rocblas_datatype_f16_r,
      ldb,
      beta_ptr,
      c,
      c_type,
      ldc,
      c,
      d_type,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      flag)));
#else
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  if (prop->major >= 7 && at::globalContext().allowFP16AccumulationCuBLAS()) {
    compute_type = CUDA_R_16F;
    halpha = alpha;
    hbeta = beta;
    alpha_ptr = &halpha;
    beta_ptr = &hbeta;
  }
  if (prop->major >= 5) {
    cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
    auto fp16_reduction = at::globalContext().allowFP16ReductionCuBLAS();
    TORCH_CHECK(fp16_reduction !=
        at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
          "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction("
          "..., allow_splitk=False) requires the cuBLASLt backend");
    if (fp16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      cublas_flags = static_cast<cublasMath_t>(
          cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
    // Disallow fp16 reductions that could lead to unexpected overflow issues.
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        alpha_ptr,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        beta_ptr,
        c,
        std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16F,
        ldc,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16F,
        ldc));
  }
#endif
}

template <typename C_Dtype>
inline void gemm_internal_cublas_bfloat16_helper(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, C_Dtype)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_SM_CARVEOUT_GUARD(handle);
  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);
  float falpha = alpha;
  float fbeta = beta;
  detail::cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(at::BFloat16);
#ifndef USE_ROCM
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  auto bf16_reduction = at::globalContext().allowBF16ReductionCuBLAS();
  TORCH_CHECK(bf16_reduction !=
      at::CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK,
        "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction("
        "..., allow_splitk=False) requires the cuBLASLt backend");
  if (bf16_reduction !=
      at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
    cublas_flags = static_cast<cublasMath_t>(
        cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }
#endif
#if defined(USE_ROCM)
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &falpha,
      a,
      CUDA_R_16BF,
      lda,
      b,
      CUDA_R_16BF,
      ldb,
      &fbeta,
      c,
      std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16BF,
      ldc,
      compute_type,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

template <>
void gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  gemm_internal_cublas_half_helper<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
}

template <>
void gemm_internal_cublas<at::Half, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  gemm_internal_cublas_half_helper<float>(CUDABLAS_GEMM_ARGS(at::Half));
}

template <>
void gemm_internal_cublas<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  gemm_internal_cublas_bfloat16_helper<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

template <>
void gemm_internal_cublas<at::BFloat16, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  gemm_internal_cublas_bfloat16_helper<float>(CUDABLAS_GEMM_ARGS(at::BFloat16));
}

template <typename Dtype, typename C_Dtype = Dtype>
inline void gemm_internal_cublaslt(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(Dtype, C_Dtype)) {
  // forward to bgemm implementation but set strides and batches to 0
  if (!bgemm_internal_cublaslt(transa, transb, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 0)) {
    gemm_internal_cublas(CUDABLAS_GEMM_ARGS(Dtype));
  }
}

template <>
void gemm_internal<double>(CUDABLAS_GEMM_ARGTYPES(double))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support double gemm yet
    gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGS(double));
#else
    gemm_internal_cublaslt<double>(CUDABLAS_GEMM_ARGS(double));
#endif
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::gemm_internal_ck<double>(CUDABLAS_GEMM_ARGS(double));
  }
#endif
  else {
    gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGS(double));
  }
}

template <>
void gemm_internal<float>(CUDABLAS_GEMM_ARGTYPES(float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<float>(CUDABLAS_GEMM_ARGS(float));
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    if (at::detail::getCUDAHooks().isGPUArch({"gfx11", "gfx12"})) { //no CK GEMM version
      gemm_internal_cublaslt<float>(CUDABLAS_GEMM_ARGS(float));
    } else{
      at::native::gemm_internal_ck<float>(CUDABLAS_GEMM_ARGS(float));
    }
  }
#endif
  else {
    gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGS(float));
  }
}

template <>
void gemm_internal<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex gemm yet
    gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
#else
    gemm_internal_cublaslt<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
#endif
  }
  else {
    gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
  }
}

template <>
void gemm_internal<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt does not support complex gemm yet
    gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
#else
    gemm_internal_cublaslt<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
#endif
  }
  else {
    gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
  }
}

template <>
void gemm_internal<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half))
{
  auto preferred = at::globalContext().blasPreferredBackend();
#ifdef USE_ROCM
  // hipBLASLt has no equivalent of rocblas_gemm_flags_fp16_alt_impl on gfx90a,
  // so a backward fp16 GEMM on the hipBLASLt path silently flushes subnormals
  // (issue #182952). Route fp16 backward GEMMs to rocBLAS so the
  // ROCmBackwardPassGuard / fp16_alt_impl handling still applies.
  if (preferred == BlasBackend::Cublaslt &&
      at::ROCmBackwardPassGuard::is_backward_pass() &&
      at::detail::getCUDAHooks().isGPUArch({"gfx90a"})) {
    preferred = BlasBackend::Cublas;
  }
#endif
  if (preferred == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (preferred == BlasBackend::Ck) {
    at::native::gemm_internal_ck<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
#endif
  else {
    gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm_internal<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    at::native::gemm_internal_ck<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
#endif
  else {
    gemm_internal_cublas<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
}

template<>
void gemm_internal<at::Half, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float))
{
  if (at::globalContext().allowFP16AccumulationCuBLAS()) {
    // Do not allow fp16 reductions with fp32 output
    TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported with allowFP16AccumulationCuBLAS");
  }

  auto preferred = at::globalContext().blasPreferredBackend();
#ifdef USE_ROCM
  // See gemm_internal<at::Half>: reroute fp16 backward GEMMs off hipBLASLt on
  // gfx90a to preserve subnormals (issue #182952).
  if (preferred == BlasBackend::Cublaslt &&
      at::ROCmBackwardPassGuard::is_backward_pass() &&
      at::detail::getCUDAHooks().isGPUArch({"gfx90a"})) {
    preferred = BlasBackend::Cublas;
  }
#endif
  if (preferred == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<at::Half, float>(CUDABLAS_GEMM_ARGS(at::Half));
  }
#if defined(USE_ROCM) && !defined(_MSC_VER)
  else if (preferred == BlasBackend::Ck) {
    TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported for ROCm");
  }
#endif
  else {
    gemm_internal_cublas<at::Half, float>(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

template<>
void gemm_internal<at::BFloat16, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float))
{
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    gemm_internal_cublaslt<at::BFloat16, float>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
#if defined(USE_ROCM) && !defined(_MSC_VER)
  else if (at::globalContext().blasPreferredBackend() == BlasBackend::Ck) {
    TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported for ROCm");
  }
#endif
  else {
    gemm_internal_cublas<at::BFloat16, float>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
}

template <typename DType, typename C_Dtype>
inline void gemm_tunable(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(DType, C_Dtype)) {
  tunable::GemmParams<DType> params;
  params.transa = transa;
  params.transb = transb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = alpha;
  params.a = a;
  params.lda = lda;
  params.b = b;
  params.ldb = ldb;
  params.beta = beta;
  params.c = c;
  params.ldc = ldc;

  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  if (transa_ && transb_) {
    static tunable::GemmTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::T> gemm{};
    gemm(&params);
  }
  else if (transa_ && !transb_) {
    static tunable::GemmTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::N> gemm{};
    gemm(&params);
  }
  else if (!transa_ && transb_) {
    static tunable::GemmTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::T> gemm{};
    gemm(&params);
  }
  else if (!transa_ && !transb_) {
    static tunable::GemmTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::N> gemm{};
    gemm(&params);
  }
  else {
    TORCH_CHECK(false, "unreachable");
  }
}

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<double>(CUDABLAS_GEMM_ARGS(double));
  }
  else {
    gemm_internal<double>(CUDABLAS_GEMM_ARGS(double));
  }
}

template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<float>(CUDABLAS_GEMM_ARGS(float));
  }
  else {
    gemm_internal<float>(CUDABLAS_GEMM_ARGS(float));
  }
}

template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
  }
  else {
    gemm_internal<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
  }
}

template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
  }
  else {
    gemm_internal<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
  }
}

template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
  else {
    gemm_internal<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
  else {
    gemm_internal<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
}

template <>
void gemm<at::Half, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::Half, float)) {
  // TODO: Support Tuning for fp16-fp32 gemm
  gemm_internal<at::Half, float>(CUDABLAS_GEMM_ARGS(at::Half));
}


template <>
void gemm<at::BFloat16, float>(CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE(at::BFloat16, float)) {
  #ifndef USE_ROCM
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();

    if (prop->major < 8)
      TORCH_CHECK(false, "gemm input type at::BFloat16 and output type float is only supported for CUDA devices with compute capability 8.0 or higher");
  #endif
  // TODO: Support Tuning for bf16-fp32 gemm
  gemm_internal<at::BFloat16, float>(CUDABLAS_GEMM_ARGS(at::BFloat16));
}


template <typename Dtype, typename C_Dtype>
bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<Dtype> alpha_val,
    const Dtype* mat1_ptr,
    int64_t mat1_ld,
    const Dtype* mat2_ptr,
    int64_t mat2_ld,
    const Dtype* bias,
    C_Dtype* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation) {

  if (std::is_same_v<C_Dtype, float> && std::is_same_v<Dtype, at::BFloat16>) {
    #ifdef USE_ROCM
    TORCH_CHECK(false, "gemm input type at::BFloat16 and output type float is not supported for ROCm");
    #endif
  } else if (std::is_same_v<C_Dtype, float> && std::is_same_v<Dtype, at::Half>) {
    #ifdef USE_ROCM
    TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported for ROCm");
    #endif
    if (at::globalContext().allowFP16AccumulationCuBLAS())
      TORCH_CHECK(false, "gemm input type at::Half and output type float is not supported with allowFP16AccumulationCuBLAS");
  }

  using opmath_t = at::opmath_type<Dtype>;
  opmath_t beta_val = bias ? 0 : 1; // bias is added in epilogue unless nullptr

  const auto type_info = detail::getCublasLtTypeInfo<Dtype, C_Dtype>();
  const cudaDataType_t abType = type_info.ab_type;
  const cudaDataType_t cType = type_info.c_type;
  const cublasComputeType_t computeType = type_info.compute_type;
  const cudaDataType_t scaleType = type_info.scale_type;
  CuBlasLtMatmulPreference preference;
  void * alpha_ptr = &alpha_val;
  void * beta_ptr = &beta_val;
#ifndef USE_ROCM
  at::Half halpha_val;
  at::Half hbeta_val;
#endif
  if constexpr (std::is_same_v<Dtype, at::Half>) {
#ifndef USE_ROCM
    if (computeType == CUBLAS_COMPUTE_16F) {
      halpha_val = alpha_val;
      hbeta_val = beta_val;
      alpha_ptr = &halpha_val;
      beta_ptr = &hbeta_val;
    }
#endif
  }
#ifndef USE_ROCM
  if constexpr (std::is_same_v<Dtype, at::Half>) {
    auto fp16_reduction = at::globalContext().allowFP16ReductionCuBLAS();
    if (fp16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      uint32_t mask =
          fp16_reduction ==
                  at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
              ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                 CUBLASLT_REDUCTION_SCHEME_NONE)
              : CUBLASLT_REDUCTION_SCHEME_NONE;
      preference.setAttribute(
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
    }
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    auto bf16_reduction = at::globalContext().allowBF16ReductionCuBLAS();
    if (bf16_reduction !=
        at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
      uint32_t mask =
          bf16_reduction ==
                  at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK
              ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                 CUBLASLT_REDUCTION_SCHEME_NONE)
              : CUBLASLT_REDUCTION_SCHEME_NONE;
      preference.setAttribute(
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, mask);
    }
  }
#endif

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);
  auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    computeDesc.setAttribute<int32_t>(
        CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
            at::globalContext()._SMCarveout_EXPERIMENTAL().value());
  }
#else
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    stream = _getCarveoutStream(
        at::globalContext()._SMCarveout_EXPERIMENTAL().value());
    _syncCurrentWithCarveoutStream(stream, true);
  }
#endif
  const cublasLtEpilogue_t epilogue =
      detail::cublasLtEpilogue(activation, bias);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);

  if (bias) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
  }

  CuBlasLtMatrixLayout Adesc(abType, m, k, mat1_ld, transpose_mat1);
  CuBlasLtMatrixLayout Bdesc(abType, k, n, mat2_ld, transpose_mat2);
  CuBlasLtMatrixLayout Cdesc(cType, m, n, result_ld);

  auto ltworkspace = CublasLtWorkspace();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ltworkspace.size);

#ifndef USE_ROCM
  uint32_t a_alignment = detail::getAlignment(reinterpret_cast<uintptr_t>(mat1_ptr));
  uint32_t b_alignment = detail::getAlignment(reinterpret_cast<uintptr_t>(mat2_ptr));
  uint32_t c_alignment = detail::getAlignment(reinterpret_cast<uintptr_t>(result_ptr));
  uint32_t d_alignment = detail::getAlignment(reinterpret_cast<uintptr_t>(bias));
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, d_alignment);
#endif

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  cublasStatus_t cublasStatus = CUBLAS_STATUS_SUCCESS;
  if (returnedResult == 0) {
    cublasStatus = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  else {
    cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      alpha_ptr,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      beta_ptr,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      ltworkspace.ptr,
      ltworkspace.size,
      stream);
#ifdef USE_ROCM
    if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
      _syncCurrentWithCarveoutStream(stream, false);
    }
#endif
  }
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    TORCH_WARN(
      "gemm_and_bias error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType,
      ". Will attempt to recover by calling unfused cublas path.");
    return false;
  }
  return true;
}

template bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<double> alpha_val,
    const double* mat1_ptr,
    int64_t mat1_ld,
    const double* mat2_ptr,
    int64_t mat2_ld,
    const double* bias,
    double* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<float> alpha_val,
    const float* mat1_ptr,
    int64_t mat1_ld,
    const float* mat2_ptr,
    int64_t mat2_ld,
    const float* bias,
    float* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::Half> alpha_val,
    const at::Half* mat1_ptr,
    int64_t mat1_ld,
    const at::Half* mat2_ptr,
    int64_t mat2_ld,
    const at::Half* bias,
    at::Half* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::Half> alpha_val,
    const at::Half* mat1_ptr,
    int64_t mat1_ld,
    const at::Half* mat2_ptr,
    int64_t mat2_ld,
    const at::Half* bias,
    float* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::BFloat16> alpha_val,
    const at::BFloat16* mat1_ptr,
    int64_t mat1_ld,
    const at::BFloat16* mat2_ptr,
    int64_t mat2_ld,
    const at::BFloat16* bias,
    at::BFloat16* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

template bool gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<at::BFloat16> alpha_val,
    const at::BFloat16* mat1_ptr,
    int64_t mat1_ld,
    const at::BFloat16* mat2_ptr,
    int64_t mat2_ld,
    const at::BFloat16* bias,
    float* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

using at::blas::ScalingType;

void scaled_gemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const void* mat1_ptr,
    const void* mat1_scale_ptr,
    int64_t mat1_ld,
    ScalarType mat1_dtype,
    ScalarType mat1_scale_dtype,
    ScalingType mat1_scaling_type,
    const void* mat2_ptr,
    const void* mat2_scale_ptr,
    int64_t mat2_ld,
    ScalarType mat2_dtype,
    ScalarType mat2_scale_dtype,
    ScalingType mat2_scaling_type,
    const void* bias_ptr,
    ScalarType bias_dtype,
    void* result_ptr,
    const void *result_scale_ptr,
    int64_t result_ld,
    ScalarType result_dtype,
    bool use_fast_accum,
    const std::optional<Tensor>& alpha) {
  // Note: see `cublasCommonArgs` for various non-intuitive manipulations
  // of input arguments to this function.
  const auto computeType = CUBLAS_COMPUTE_32F;
  const auto scaleType = CUDA_R_32F;
  // Note: alpha_val may change later depending on user-passed argument
  float alpha_val = 1.0;
  float beta_val = 0.0;
#ifndef USE_ROCM
  // Note: unused, but cublasLtMatmul requires a C pointer that is not result_ptr or nullptr
  const void* dummy_C_ptr = mat1_ptr;
#endif // ifndef USE_ROCM
  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, detail::cublasOpFromChar(transa));
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, detail::cublasOpFromChar(transb));
  cublasLtMatmulDescAttributes_t matmulDescA = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER;
  cublasLtMatmulDescAttributes_t matmulDescB = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER;
#if defined(USE_ROCM) && !defined(HIPBLASLT_OUTER_VEC) && defined(HIPBLASLT_VEC_EXT)
  // hipblaslt supported row-wise before cublas, and did so their own way (via
  // the SCALE_POINTERSs), but then migrated to match how cublas does it (via
  // the SCALE_MODEs). Here we check for this early custom mode.
  bool use_rowwise = (mat1_scaling_type == ScalingType::RowWise && mat2_scaling_type == ScalingType::RowWise);
  if (use_rowwise) {
    matmulDescA = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT;
    matmulDescB = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT;
  }
  else if (mat1_scale_dtype == kFloat8_e8m0fnu && mat2_scale_dtype == kFloat8_e8m0fnu) {
  #if ROCM_VERSION >= 70000
            std::vector<std::string> mx_archs{"gfx950"};
  #if ROCM_VERSION >= 71400
            mx_archs.push_back("gfx1250");
  #endif
            if (at::detail::getCUDAHooks().isGPUArch(mx_archs)) {
                // TODO: add constraints based on hipblaslt internals
                TORCH_CHECK((m % 16 == 0) && (n % 16 == 0) && (k % 128 == 0),
                           "M, N must be multiples of 16 and K should be multiple of 128 for MX format. "
                           "Got m=", m, ", n=", n, ", k=", k);
            }
  #endif
  }
#elif (CUDA_VERSION < 12090) && !defined(USE_ROCM)
  // hipblaslt supported row-wise before cublas, and did so their own way (via
  // the SCALE_POINTERSs), but then migrated to match how cublas does it (via
  // the SCALE_MODEs). Here we check for this early custom mode.
  bool use_rowwise = (mat1_scaling_type == ScalingType::RowWise && mat2_scaling_type == ScalingType::RowWise);
  // rowwise isn't supported using older cublaslt or older hipblaslt
  TORCH_INTERNAL_ASSERT(use_rowwise == false, "rowwise scaled_gemm not supported with blaslt");
#endif  // if defined(USE_ROCM) && !defined(HIPBLASLT_OUTER_VEC) && defined(HIPBLASLT_VEC_EXT)
  computeDesc.setAttribute(matmulDescA, mat1_scale_ptr);
  computeDesc.setAttribute(matmulDescB, mat2_scale_ptr);
  if (result_scale_ptr != nullptr) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, result_scale_ptr);
  }
  auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    computeDesc.setAttribute<int32_t>(
        CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
            at::globalContext()._SMCarveout_EXPERIMENTAL().value());
  }
#else
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    stream = _getCarveoutStream(
        at::globalContext()._SMCarveout_EXPERIMENTAL().value());
    _syncCurrentWithCarveoutStream(stream, true);
  }
#endif // ifndef USE_ROCM
#ifndef USE_ROCM
  const int8_t fastAccuMode = use_fast_accum ? 1 : 0;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fastAccuMode);
#endif // ifndef USE_ROCM
  CuBlasLtMatrixLayout Adesc(ScalarTypeToCudaDataType(mat1_dtype), m, k, mat1_ld, transa == 't');
  CuBlasLtMatrixLayout Bdesc(ScalarTypeToCudaDataType(mat2_dtype), k, n, mat2_ld, transb == 't');
#ifdef USE_ROCM
  // Cdesc is unused, beta is 0. But hipblaslt needs this set to something reasonable.
  CuBlasLtMatrixLayout Cdesc(ScalarTypeToCudaDataType(result_dtype), m, n, result_ld);
#else
  CuBlasLtMatrixLayout Cdesc(ScalarTypeToCudaDataType(bias_dtype), m, n, result_ld);
#endif // ifdef USE_ROCM
  CuBlasLtMatrixLayout Ddesc(ScalarTypeToCudaDataType(result_dtype), m, n, result_ld);
  if (bias_ptr) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, ScalarTypeToCudaDataType(bias_dtype));
  }

  // Handle user-passed alpha
  float *alpha_ptr = &alpha_val;
  float *beta_ptr = &beta_val;

  if (alpha.has_value()) {
    auto& a = alpha.value();

    // if device-tensor
    if (a.is_cuda()) {
      // NOTE: there are lifetime requirements on device-side pointers for alpha/beta -- the value must be
      //       valid & correct until the cublas call finishes (not is scheduled like host-side values). Thus
      //       we need to use allocations for alpha/beta that have some guarantees on lifetime - a statically
      //       managed 4B buffer for alpha that we'll copy the passed alpha value into, and constant memory
      //       for beta respectively.
      float *user_alpha_ptr = at::cuda::detail::get_user_alpha_ptr();
      at::Tensor user_alpha = at::from_blob(user_alpha_ptr, {1}, TensorOptions().device(kCUDA).dtype(kFloat));
      user_alpha.copy_(a);
      // Tell cublasLt we're using device-side pointers for alpha/beta
      auto pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
      computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_POINTER_MODE, pointer_mode);
      alpha_ptr = user_alpha.data_ptr<float>();
      beta_ptr = at::cuda::detail::get_cublas_device_zero();
    } else {
      alpha_val = a.item<float>();
    }
  }
    // For other data types, use the get_scale_mode function based on scaling type
    // The SCALE_MODE attrs only exist in cuBLAS 12.8+/ROCm 7.0 or in recent hipblaslt,
    // but we must invoke get_scale_mode anyways to trigger the version checks.
    // Note that AMD/ROCm follows OCP Spec 1.0, which is different from NVIDIA's implementation. See get_scale_mode() for details.
    [[maybe_unused]] int a_scale_mode = detail::cublasLtMatmulScaleMode(mat1_scaling_type, mat1_scale_dtype, use_fast_accum);
    [[maybe_unused]] int b_scale_mode = detail::cublasLtMatmulScaleMode(mat2_scaling_type, mat2_scale_dtype, use_fast_accum);
#if CUDA_VERSION >= 12080 || (defined(USE_ROCM) && ROCM_VERSION >= 70000 && defined(HIPBLASLT_OUTER_VEC))
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, a_scale_mode);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, b_scale_mode);
#endif // if CUDA_VERSION >= 12080 || (defined(USE_ROCM) && ROCM_VERSION >= 70000 && defined(HIPBLASLT_OUTER_VEC))

  CuBlasLtMatmulPreference preference;
  auto ltworkspace = CublasLtWorkspace();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ltworkspace.size);
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Ddesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
#ifndef USE_ROCM
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
#else
    // hipblaslt might be able to recover by returning all algos
    std::vector<hipblasLtMatmulHeuristicResult_t> all_algos;
    TORCH_CUDABLAS_CHECK(hipblaslt_ext::getAllAlgos(
        ltHandle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        detail::cublasOpFromChar(transa),
        detail::cublasOpFromChar(transb),
        ScalarTypeToCudaDataType(mat1_dtype),
        ScalarTypeToCudaDataType(mat2_dtype),
        // C is nullptr and beta=0, so set to something reasonable. See above.
        //ScalarTypeToCudaDataType(bias_dtype),
        ScalarTypeToCudaDataType(result_dtype),
        ScalarTypeToCudaDataType(result_dtype),
        CUBLAS_COMPUTE_32F,
        all_algos));
    if (all_algos.size() == 0) {
      TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    // pick first valid solution
    bool found = false;
    for (size_t i = 0; i < all_algos.size(); i++) {
        size_t ret_workspace_size = 0;
        auto is_valid_status = hipblaslt_ext::matmulIsAlgoSupported(
                ltHandle,
                computeDesc.descriptor(),
                alpha_ptr,
                Adesc.descriptor(),
                Bdesc.descriptor(),
                beta_ptr,
                Cdesc.descriptor(),
                Ddesc.descriptor(),
                all_algos[i].algo,
                ret_workspace_size);
        if (is_valid_status == HIPBLAS_STATUS_SUCCESS) {
            if (ret_workspace_size <= ltworkspace.size) {
                heuristicResult = all_algos[i];
                found = true;
                break;
            }
        }
    }
    TORCH_CHECK(found, "could not find valid hipblaslt solution");
#endif // ifndef USE_ROCM
  }
  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      alpha_ptr,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      beta_ptr,
#ifdef USE_ROCM
      result_ptr, // unused, since beta_val is 0, but hipblaslt can't handle nullptr
#else
      dummy_C_ptr, // also unused, but cuBLAS can't use nullptr or result_ptr
#endif // ifdef USE_ROCM
      Cdesc.descriptor(),
      result_ptr,
      Ddesc.descriptor(),
      &heuristicResult.algo,
      ltworkspace.ptr,
      ltworkspace.size,
      stream);
#ifdef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    _syncCurrentWithCarveoutStream(stream, false);
  }
#endif
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transa,
      " transpose_mat2 ",
      transb,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
  return;
}

void int8_gemm(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    const int8_t* mat1_ptr,
    int64_t mat1_ld,
    const int8_t* mat2_ptr,
    int64_t mat2_ld,
    int32_t* result_ptr,
    int64_t result_ld) {

  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
  cudaDataType_t scaleType = CUDA_R_32I;

  cudaDataType_t abType = CUDA_R_8I;
  cudaDataType_t cType = CUDA_R_32I;

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);
  auto stream = at::cuda::getCurrentCUDAStream();
#ifndef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    computeDesc.setAttribute<int32_t>(
        CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
            at::globalContext()._SMCarveout_EXPERIMENTAL().value());
  }
#else
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    stream = _getCarveoutStream(
        at::globalContext()._SMCarveout_EXPERIMENTAL().value());
    _syncCurrentWithCarveoutStream(stream, true);
  }
#endif

  CuBlasLtMatrixLayout Adesc(abType, m, k, mat1_ld, transpose_mat1);
  CuBlasLtMatrixLayout Bdesc(abType, k, n, mat2_ld, transpose_mat2);
  CuBlasLtMatrixLayout Cdesc(cType, m, n, result_ld);

  // cublas team: alpha and beta need to be the same dtype as of scaleType
  at::opmath_type<int32_t> alpha_val = 1;
  int32_t beta_val = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

#ifdef USE_ROCM
  CuBlasLtMatmulPreference preference;
  auto ltworkspace = CublasLtWorkspace();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ltworkspace.size);
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }
#endif

  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
#ifdef USE_ROCM
      &heuristicResult.algo,
#else
      nullptr, // Heuristics don't seem to work for int8
#endif
#ifdef USE_ROCM
      ltworkspace.ptr,
#else
      nullptr, // Non-zero workspace doesn't seem to work.
#endif
#ifdef USE_ROCM
      ltworkspace.size,
#else
      0,
#endif
      stream);
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
#ifdef USE_ROCM
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    _syncCurrentWithCarveoutStream(stream, false);
  }
#endif
}

void grouped_gemm(
    char transa,
    char transb,
    const void *mArrayDev,
    int64_t avgM,
    const void *nArrayDev,
    int64_t avgN,
    const void *kArrayDev,
    int64_t avgK,
    const int64_t *alphaArrayDev,
    const float *alphaScalar,
    ScalarType input_dtype,
    const int64_t *APtrArrayDev,
    const void *ldaArrayDev,
    const int64_t *BPtrArrayDev,
    const void *ldbArrayDev,
    const int64_t *betaArrayDev,
    const float *betaScalar,
    ScalarType result_dtype,
    const int64_t *CPtrArrayDev,
    const void *ldcArrayDev,
    int64_t *DPtrArrayDev,
    const void *lddArrayDev,
    int batchCount,
    bool use_int64_dims) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  const bool sm90 = prop->major == 9;
  TORCH_CHECK(prop->major >= 9 && prop->major < 12, "grouped cublasLtMatmul requires SM 9.0-11.0");

  const auto computeType = CUBLAS_COMPUTE_32F;
  const auto scaleType = CUDA_R_32F;
  const auto pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  const int64_t alphaBatchStride = sm90 ? 0 : 1;
  const int64_t betaBatchStride = sm90 ? 0 : 1;

  cublasOperation_t opa = detail::cublasOpFromChar(transa);
  cublasOperation_t opb = detail::cublasOpFromChar(transb);

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_POINTER_MODE, pointer_mode);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, alphaBatchStride);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE, betaBatchStride);

  CuBlasLtGroupedMatrixLayout Adesc(ScalarTypeToCudaDataType(input_dtype), batchCount, mArrayDev, kArrayDev, ldaArrayDev, opa != CUBLAS_OP_N, use_int64_dims);
  CuBlasLtGroupedMatrixLayout Bdesc(ScalarTypeToCudaDataType(input_dtype), batchCount, kArrayDev, nArrayDev, ldbArrayDev, opb != CUBLAS_OP_N, use_int64_dims);
  CuBlasLtGroupedMatrixLayout Cdesc(ScalarTypeToCudaDataType(result_dtype), batchCount, mArrayDev, nArrayDev, ldcArrayDev, false, use_int64_dims);
  CuBlasLtGroupedMatrixLayout Ddesc(ScalarTypeToCudaDataType(result_dtype), batchCount, mArrayDev, nArrayDev, lddArrayDev, false, use_int64_dims);

  CuBlasLtMatmulPreference preference;
  auto ltworkspace = CublasLtWorkspace();
  auto stream = at::cuda::getCurrentCUDAStream();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ltworkspace.size);
  // As of CUDA 13.3u1, cuBLASLt grouped GEMM heuristics return suboptimal algos when inputs are row major
  // Swap average rows and cols to recover optimal kernels, and update this once fixed in cuBLASLt
  preference.setAttribute(CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, avgN);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, avgM);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, avgK);

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Ddesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  // When alphaBatchStride=0 (SM 9.0), alpha/beta are single device scalars.
  // When alphaBatchStride=1 (SM 10.0+), alpha/beta are per-group pointer arrays.
  const void* alpha = alphaBatchStride ? static_cast<const void*>(alphaArrayDev) : static_cast<const void*>(alphaScalar);
  const void* beta = betaBatchStride ? static_cast<const void*>(betaArrayDev) : static_cast<const void*>(betaScalar);

  cublasStatus_t cublasStatus = cublasLtMatmul(
    ltHandle,
    computeDesc.descriptor(),
    alpha,
    APtrArrayDev,
    Adesc.descriptor(),
    BPtrArrayDev,
    Bdesc.descriptor(),
    beta,
    CPtrArrayDev,
    Cdesc.descriptor(),
    DPtrArrayDev,
    Ddesc.descriptor(),
    &heuristicResult.algo,
    ltworkspace.ptr,
    ltworkspace.size,
    stream);

  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling grouped cublasLtMatmul");
  return;
#else
  TORCH_CHECK(false, "grouped cublasLtMatmul requires CUDA >= 13.2 and is not supported on ROCm. Current build does not meet these requirements.");
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
}

template <>
void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasStrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDtrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<c10::complex<float>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      reinterpret_cast<const cuComplex*>(A),
      lda,
      reinterpret_cast<cuComplex*>(B),
      ldb));
}

template <>
void trsm<c10::complex<double>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      reinterpret_cast<const cuDoubleComplex*>(A),
      lda,
      reinterpret_cast<cuDoubleComplex*>(B),
      ldb));
}

template <>
// NOLINTNEXTLINE(*array*)
void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasStrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
// NOLINTNEXTLINE(*array*)
void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<float>>(
// NOLINTNEXTLINE(*array*)
    CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      reinterpret_cast<cuComplex**>(A),
      lda,
      reinterpret_cast<cuComplex**>(B),
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<double>>(
// NOLINTNEXTLINE(*array*)
    CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      reinterpret_cast<cuDoubleComplex**>(A),
      lda,
      reinterpret_cast<cuDoubleComplex**>(B),
      ldb,
      batchCount));
}

/* LEVEL 2 BLAS FUNCTIONS */

#define GEMV_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, n); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incx); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incy); \
  } while (0)

template <>
void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = detail::cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_CUDABLAS_CHECK(
      cublasZgemv(handle, op, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(y), incy));
}

template <>
void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>)) {
  // gemv is bw bound, and does not benefit from TF32. But the precision
  // loss still happens on TF32. So we disable it here.
  NoTF32Guard disable_tf32;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = detail::cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_CUDABLAS_CHECK(
      cublasCgemv(handle, op, m, n, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(y), incy));
}

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = detail::cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(double);
  TORCH_CUDABLAS_CHECK(
      cublasDgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float)) {
  // gemv is bw bound, and does not benefit from TF32. But the precision
  // loss still happens on TF32. So we disable it here.
  NoTF32Guard disable_tf32;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t op = detail::cublasOpFromChar(trans);
  _cublasAdjustLdLevel2(m, n, &lda);
  GEMV_CHECK_ARGVALUES(float);
  TORCH_CUDABLAS_CHECK(
      cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half)) {
  // In general, cublas regards matrices as column-major.
  // The cublasS/Dgemv usages in cuda::blas::gemv<float>/<double> above
  // require that external blas::gemv callers obey the following convention:
  //
  // If "a" is row-major with shape (output, summed) in blas::gemv's caller,
  // caller interprets it as column-major with shape (summed, output), passes
  // summed and output respectively to our local vars m, n, and requests that cublas
  // internally transpose ("trans") the column-major interpretation of a.
  //
  // There's no such thing as "cublasHalfgemv", so here we hack gemv with a gemm.
  // However, we must allow the same calling convention, because the caller shouldn't
  // have to swap args based on whether it's calling blas::gemv<at::Half> or <float>.

  bool trans_bool = (detail::cublasOpFromChar(trans) != CUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  // After swap, local vars m, n contain the output and summed sizes respectively,
  // regardless of whether "a" was row-major or column-major in gemv<>'s caller.

  // To handle the possibility incy > 1, interprets vector y as column-major matrix with one row
  // (shape (1, output)) and leading dim incy.
  // trans(a)*x would compute a matrix with one column (shape (output, 1)) which wouldn't match y.
  // So instead, we interpret x similarly to y, as a column-major matrix with one row
  // (shape (1, summed)) and leading dim incx.  The gemm then carries out x*transpose(trans(a)) to
  // produce a matrix with one row (shape (1, output)), matching y.
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::Half>(
      'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}

template <>
void gemv<at::BFloat16>(CUDABLAS_GEMV_ARGTYPES(at::BFloat16)) {
  bool trans_bool = (detail::cublasOpFromChar(trans) != CUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  char trans_flipped = (trans_bool ? 'n' : 't');
  gemm<at::BFloat16>(
      'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
}

/* LEVEL 1 BLAS FUNCTIONS */

template <>
void dot<double>(CUDABLAS_DOT_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
}

template <>
void dot<float>(CUDABLAS_DOT_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
}

template <>
void dot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZdotu(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

template <>
void dot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCdotu(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

template <>
void dot<at::Half>(CUDABLAS_DOT_ARGTYPES(at::Half)) {
  TORCH_CUDABLAS_CHECK(cublasDotEx(
      handle,
      n,
      x,
      CUDA_R_16F,
      incx,
      y,
      CUDA_R_16F,
      incy,
      result,
      CUDA_R_16F,
      CUDA_R_32F));
}

template <>
void dot<at::BFloat16>(CUDABLAS_DOT_ARGTYPES(at::BFloat16)) {
  TORCH_CUDABLAS_CHECK(cublasDotEx(
      handle,
      n,
      x,
      CUDA_R_16BF,
      incx,
      y,
      CUDA_R_16BF,
      incy,
      result,
      CUDA_R_16BF,
      CUDA_R_32F));
}

template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

template <>
void getrsBatched<float>(CUDABLAS_GETRS_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      dA_array,
      lda,
      ipiv_array,
      dB_array,
      ldb,
      info_array,
      batchsize));
}

template <>
void getrsBatched<double>(CUDABLAS_GETRS_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      dA_array,
      lda,
      ipiv_array,
      dB_array,
      ldb,
      info_array,
      batchsize));
}

template <>
void getrsBatched<c10::complex<float>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<cuComplex**>(dA_array),
      lda,
      ipiv_array,
      reinterpret_cast<cuComplex**>(dB_array),
      ldb,
      info_array,
      batchsize));
}

template <>
void getrsBatched<c10::complex<double>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<cuDoubleComplex**>(dA_array),
      lda,
      ipiv_array,
      reinterpret_cast<cuDoubleComplex**>(dB_array),
      ldb,
      info_array,
      batchsize));
}

template <>
void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSgeqrfBatched(
      handle, m, n, A_array, lda, tau_array, info, batchsize));
}

template <>
void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDgeqrfBatched(
      handle, m, n, A_array, lda, tau_array, info, batchsize));
}

template <>
void geqrfBatched<c10::complex<float>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCgeqrfBatched(
      handle,
      m,
      n,
      reinterpret_cast<cuComplex**>(A_array),
      lda,
      reinterpret_cast<cuComplex**>(tau_array),
      info,
      batchsize));
}

template <>
void geqrfBatched<c10::complex<double>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZgeqrfBatched(
      handle,
      m,
      n,
      reinterpret_cast<cuDoubleComplex**>(A_array),
      lda,
      reinterpret_cast<cuDoubleComplex**>(tau_array),
      info,
      batchsize));
}

template <>
void getrfBatched<double>(
    int n, double** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasDgetrfBatched(
      handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
}

template <>
void getrfBatched<float>(
    int n, float** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSgetrfBatched(
      handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
}

template <>
void getrfBatched<c10::complex<double>>(
    int n,
    c10::complex<double>** dA_array,
    int ldda,
    int* ipiv_array,
    int* info_array,
    int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasZgetrfBatched(
      handle,
      n,
      reinterpret_cast<cuDoubleComplex**>(dA_array),
      ldda,
      ipiv_array,
      info_array,
      batchsize));
}

template <>
void getrfBatched<c10::complex<float>>(
    int n,
    c10::complex<float>** dA_array,
    int ldda,
    int* ipiv_array,
    int* info_array,
    int batchsize) {
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasCgetrfBatched(
      handle,
      n,
      reinterpret_cast<cuComplex**>(dA_array),
      ldda,
      ipiv_array,
      info_array,
      batchsize));
}


template <>
void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double)) {
  TORCH_CUDABLAS_CHECK(cublasDgelsBatched(
      handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
}

template <>
void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float)) {
  TORCH_CUDABLAS_CHECK(cublasSgelsBatched(
      handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
}

template <>
void gelsBatched<c10::complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDABLAS_CHECK(cublasZgelsBatched(
      handle, trans,
      m, n, nrhs,
      reinterpret_cast<cuDoubleComplex**>(dA_array),
      ldda,
      reinterpret_cast<cuDoubleComplex**>(dC_array),
      lddc,
      info,
      devInfoArray,
      batchSize));
}

template <>
void gelsBatched<c10::complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDABLAS_CHECK(cublasCgelsBatched(
      handle, trans,
      m, n, nrhs,
      reinterpret_cast<cuComplex**>(dA_array),
      ldda,
      reinterpret_cast<cuComplex**>(dC_array),
      lddc,
      info,
      devInfoArray,
      batchSize));
}

} // namespace at::cuda::blas
