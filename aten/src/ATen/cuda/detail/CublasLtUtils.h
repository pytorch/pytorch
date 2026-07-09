#pragma once

#include <ATen/BlasBackend.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace at::cuda::blas::detail {

template <typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, cublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};

class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<
                                     cublasLtMatmulDescOpaque_t,
                                     &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(
      cublasComputeType_t compute_type,
      cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }

  template <typename T>
  void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulDescSetAttribute(
        descriptor(), attr, &value, sizeof(value)));
  }
};

class CuBlasLtMatrixLayout : public CuBlasLtDescriptor<
                                 cublasLtMatrixLayoutOpaque_t,
                                 &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(
      cudaDataType_t type,
      uint64_t rows,
      uint64_t cols,
      int64_t ld,
      bool t = false) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(
        &raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }

  template <typename T>
  void setAttribute(cublasLtMatrixLayoutAttribute_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatrixLayoutSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020
class CuBlasLtGroupedMatrixLayout : public CuBlasLtDescriptor<
                                        cublasLtMatrixLayoutOpaque_t,
                                        &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtGroupedMatrixLayout(
      cudaDataType_t type,
      int group_count,
      const void* rows_array,
      const void* cols_array,
      const void* ld_array,
      bool t = false,
      bool use_int64 = false) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
        &raw_descriptor,
        type,
        group_count,
        t ? cols_array : rows_array,
        t ? rows_array : cols_array,
        ld_array));
    descriptor_.reset(raw_descriptor);
    setAttribute(CUBLASLT_MATRIX_LAYOUT_ORDER, CUBLASLT_ORDER_ROW);
    if (use_int64) {
      setAttribute(
          CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
          CUBLASLT_INTEGER_WIDTH_64);
      setAttribute(
          CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
          CUBLASLT_INTEGER_WIDTH_64);
    }
  }

  template <typename T>
  void setAttribute(cublasLtMatrixLayoutAttribute_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatrixLayoutSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13020

class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<
                                     cublasLtMatmulPreferenceOpaque_t,
                                     &cublasLtMatmulPreferenceDestroy> {
 public:
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }

  template <typename T>
  void setAttribute(cublasLtMatmulPreferenceAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulPreferenceSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};

struct CublasLtWorkspace {
  CublasLtWorkspace() {
    size = at::cuda::getCUDABlasLtWorkspaceSize();
    ptr = at::cuda::getCUDABlasLtWorkspace();
  }

  void* ptr;
  size_t size;
};

inline cublasOperation_t cublasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return CUBLAS_OP_N;
    case 't':
    case 'T':
      return CUBLAS_OP_T;
    case 'c':
    case 'C':
      return CUBLAS_OP_C;
  }
  TORCH_CHECK(
      false,
      "cublasOpFromChar input should be 't', 'n' or 'c' but got `",
      op,
      "`");
  return CUBLAS_OP_N; // unreachable - TORCH_CHECK(false) throws
}

inline int cublasLtMatmulScaleMode(
    at::blas::ScalingType scaling_type,
    ScalarType scale_dtype,
    bool use_fast_accum) {
  switch (scaling_type) {
    case at::blas::ScalingType::BlockWise1x32:
      TORCH_CHECK(scale_dtype == kFloat8_e8m0fnu);
#if CUDA_VERSION >= 12080 || (defined(USE_ROCM) && ROCM_VERSION >= 70000)
      return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
#else
      TORCH_CHECK(
          false,
          "scaled_gemm with `torch.float8_e8m0fnu` scales of 1x32 blocks "
          "is only supported for CUDA 12.8 and above");
#endif
    case at::blas::ScalingType::BlockWise1x16:
      TORCH_CHECK(scale_dtype == kFloat8_e4m3fn);
#if CUDA_VERSION >= 12080
      return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
#else
      TORCH_CHECK(
          false,
          "scaled_gemm with `torch.float8_e4m3fn` scales of 1x16 blocks "
          "is only supported for CUDA 12.8 and above");
#endif
    case at::blas::ScalingType::RowWise:
      TORCH_CHECK(scale_dtype == kFloat);
#if CUDA_VERSION >= 12090 || (defined(USE_ROCM) && defined(HIPBLASLT_OUTER_VEC))
      return CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
#elif defined(USE_ROCM) && defined(HIPBLASLT_VEC_EXT)
      // Old hipBLASLt rowwise mode is activated through SCALE_POINTER_VEC_EXT.
      return 0;
#else
      TORCH_CHECK(
          false,
          "scaled_gemm with rowwise scaling is only supported for CUDA 12.9 "
          "and above");
#endif
    case at::blas::ScalingType::BlockWise1x128:
      TORCH_CHECK(scale_dtype == kFloat);
      TORCH_CHECK(
          !use_fast_accum,
          "scaled_gemm doesn't support fast accum with 1x128 blockwise scaling");
#if CUDA_VERSION >= 12090
      return CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
#else
      TORCH_CHECK(
          false,
          "scaled_gemm with 1x128 blockwise scaling is only supported for "
          "CUDA 12.9 and above");
#endif
    case at::blas::ScalingType::BlockWise128x128:
      TORCH_CHECK(scale_dtype == kFloat);
      TORCH_CHECK(
          !use_fast_accum,
          "scaled_gemm doesn't support fast accum with 128x128 blockwise scaling");
#if CUDA_VERSION >= 12090
      return CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
#else
      TORCH_CHECK(
          false,
          "scaled_gemm with 128x128 blockwise scaling is only supported for "
          "CUDA 12.9 and above");
#endif
    case at::blas::ScalingType::TensorWise:
      TORCH_CHECK(scale_dtype == kFloat);
#if CUDA_VERSION >= 12080
      return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
#else
      return 0;
#endif
    default:
      TORCH_CHECK(false);
  }
}

inline void cublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  if (n <= 1) {
    *ldc = std::max<int64_t>(m, 1);
  }
  if (transa_) {
    if (m <= 1) {
      *lda = std::max<int64_t>(k, 1);
    }
  } else {
    if (k <= 1) {
      *lda = std::max<int64_t>(m, 1);
    }
  }
  if (transb_) {
    if (k <= 1) {
      *ldb = std::max<int64_t>(n, 1);
    }
  } else {
    if (n <= 1) {
      *ldb = std::max<int64_t>(k, 1);
    }
  }
}

#ifndef USE_ROCM
inline uint32_t getAlignment(uintptr_t address) {
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;
    }
  }
}
#endif

template <typename T, typename C_Dtype = T>
struct CublasLtTypeInfo {
  cudaDataType_t ab_type = CUDA_R_32F;
  cudaDataType_t c_type = CUDA_R_32F;
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  cudaDataType_t scale_type = CUDA_R_32F;
};

template <typename T, typename C_Dtype = T>
CublasLtTypeInfo<T, C_Dtype> getCublasLtTypeInfo() {
  CublasLtTypeInfo<T, C_Dtype> info;
  if constexpr (std::is_same_v<T, double>) {
    info.ab_type = CUDA_R_64F;
    info.c_type = CUDA_R_64F;
    info.compute_type = CUBLAS_COMPUTE_64F;
    info.scale_type = CUDA_R_64F;
  } else if constexpr (std::is_same_v<T, float>) {
    if (at::globalContext().float32Precision(
            at::Float32Backend::CUDA,
            at::Float32Op::MATMUL) == at::Float32Precision::TF32) {
      info.compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
  } else if constexpr (std::is_same_v<T, c10::complex<double>>) {
    info.ab_type = CUDA_C_64F;
    info.c_type = CUDA_C_64F;
    info.compute_type = CUBLAS_COMPUTE_64F;
    info.scale_type = CUDA_C_64F;
  } else if constexpr (std::is_same_v<T, c10::complex<float>>) {
    info.ab_type = CUDA_C_32F;
    info.c_type = CUDA_C_32F;
    info.scale_type = CUDA_C_32F;
  } else if constexpr (std::is_same_v<T, at::Half>) {
#ifndef USE_ROCM
    auto* prop = at::cuda::getCurrentDeviceProperties();
    if (prop->major >= 7 && at::globalContext().allowFP16AccumulationCuBLAS()) {
      info.compute_type = CUBLAS_COMPUTE_16F;
      info.scale_type = CUDA_R_16F;
    }
#endif
    info.ab_type = CUDA_R_16F;
    info.c_type = std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16F;
  } else if constexpr (std::is_same_v<T, at::BFloat16>) {
    info.ab_type = CUDA_R_16BF;
    info.c_type = std::is_same_v<C_Dtype, float> ? CUDA_R_32F : CUDA_R_16BF;
  } else {
    static_assert(
        false && sizeof(T),
        "getCublasLtTypeInfo: unsupported cuBLASLt type");
  }
  return info;
}

inline cublasLtEpilogue_t cublasLtEpilogue(
    at::cuda::blas::GEMMAndBiasActivationEpilogue activation,
    const void* bias) {
  switch (activation) {
    case at::cuda::blas::GEMMAndBiasActivationEpilogue::RELU:
      return bias ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU;
    case at::cuda::blas::GEMMAndBiasActivationEpilogue::GELU:
      return bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU;
    default:
      return bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
  }
}

} // namespace at::cuda::blas::detail
