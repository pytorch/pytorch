// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/StringUtil.h>
#include <fmt/printf.h>

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#define TORCH_HIPBLASLT_CHECK(EXPR)               \
  do {                                            \
    hipblasStatus_t __err = EXPR;                 \
    TORCH_CHECK(__err == HIPBLAS_STATUS_SUCCESS,  \
                "hipblaslt error: ",              \
                hipblasStatusToString(__err),     \
                " when calling `" #EXPR "`");     \
  } while (0)

namespace at::cuda::tunable {

template <typename T>
constexpr hipDataType HipDataTypeFor();

template <>
constexpr hipDataType HipDataTypeFor<float>() {
  return HIP_R_32F;
}

template <>
constexpr hipDataType HipDataTypeFor<Half>() {
  return HIP_R_16F;
}

template <>
constexpr hipDataType HipDataTypeFor<BFloat16>() {
  return HIP_R_16BF;
}

template <>
constexpr hipDataType HipDataTypeFor<double>() {
  return HIP_R_64F;
}

template <>
constexpr hipDataType HipDataTypeFor<c10::Float8_e4m3fnuz>() {
  return HIP_R_8F_E4M3_FNUZ;
}

template <>
constexpr hipDataType HipDataTypeFor<c10::Float8_e5m2fnuz>() {
  return HIP_R_8F_E5M2_FNUZ;
}

// This code is instantiated regardless of ROCm version.
// Prior to ROCm 6.3, we hard-code the known enum values.
template <>
constexpr hipDataType HipDataTypeFor<c10::Float8_e4m3fn>() {
#if ROCM_VERSION >= 60300
  return HIP_R_8F_E4M3;
#else
  return static_cast<hipDataType>(28);
#endif
}

template <>
constexpr hipDataType HipDataTypeFor<c10::Float8_e5m2>() {
#if ROCM_VERSION >= 60300
  return HIP_R_8F_E5M2;
#else
  return static_cast<hipDataType>(29);
#endif
}

// This type is not intended for matrix types but rather a scale factor.
// Return a dummy value to satisfy linker.
template <>
constexpr hipDataType HipDataTypeFor<c10::Float8_e8m0fnu>() {
  return static_cast<hipDataType>(500);
}

template <typename T>
int GetBatchFromParams(const GemmParams<T>* params) {
  return 1;
}

template <typename T>
int GetBatchFromParams(const GemmAndBiasParams<T>* params) {
  return 1;
}

template <typename T>
int GetBatchFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->batch;
}

template <typename T>
int GetBatchFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideAFromParams(const GemmParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideAFromParams(const GemmAndBiasParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideAFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_a;
}

template <typename T>
int GetStrideAFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideBFromParams(const GemmParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideBFromParams(const GemmAndBiasParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideBFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_b;
}

template <typename T>
int GetStrideBFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideCFromParams(const GemmParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideCFromParams(const GemmAndBiasParams<T>* params) {
  return 1;
}

template <typename T>
int GetStrideCFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_c;
}

template <typename T>
int GetStrideCFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

template <typename T>
float GetAlphaFromParams(const GemmParams<T>* params) {
  return params->alpha;
}

template <typename T>
float GetAlphaFromParams(const GemmAndBiasParams<T>* params) {
  return params->alpha;
}

template <typename T>
float GetAlphaFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->alpha;
}

template <typename T>
float GetAlphaFromParams(const ScaledGemmParams<T>* params) {
  return 1.0;
}

template <typename T>
float GetBetaFromParams(const GemmParams<T>* params) {
  return params->beta;
}

template <typename T>
float GetBetaFromParams(const GemmAndBiasParams<T>* params) {
  return 0.0;
}

template <typename T>
float GetBetaFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->beta;
}

template <typename T>
float GetBetaFromParams(const ScaledGemmParams<T>* params) {
  return 0.0;
}

template <typename T>
bool GetUseRowwiseFromParams(const GemmParams<T>* params) {
  return false;
}

template <typename T>
bool GetUseRowwiseFromParams(const GemmAndBiasParams<T>* params) {
  return false;
}

template <typename T>
bool GetUseRowwiseFromParams(const GemmStridedBatchedParams<T>* params) {
  return false;
}

template <typename T>
bool GetUseRowwiseFromParams(const ScaledGemmParams<T>* params) {
  return params->use_rowwise;
}

template <typename T>
const void* GetAScalePointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetAScalePointerFromParams(const GemmAndBiasParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetAScalePointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetAScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->a_scale_ptr;
}

template <typename T>
const void* GetBScalePointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetBScalePointerFromParams(const GemmAndBiasParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetBScalePointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetBScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->b_scale_ptr;
}

template <typename T>
const void* GetDScalePointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetDScalePointerFromParams(const GemmAndBiasParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetDScalePointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetDScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->c_scale_ptr;
}

template <typename T>
const void* GetBiasPointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetBiasPointerFromParams(const GemmAndBiasParams<T>* params) {
  return params->bias;
}

template <typename T>
const void* GetBiasPointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetBiasPointerFromParams(const ScaledGemmParams<T>* params) {
  return params->bias_ptr;
}

template <typename T>
hipDataType GetBiasTypeFromParams(const GemmParams<T>* params) {
  return HIP_R_32F;
}

template <typename T>
hipDataType GetBiasTypeFromParams(const GemmAndBiasParams<T>* params) {
  return HipDataTypeFor<T>();
}

template <typename T>
hipDataType GetBiasTypeFromParams(const GemmStridedBatchedParams<T>* params) {
  return HIP_R_32F;
}

template <typename T>
hipDataType GetBiasTypeFromParams(const ScaledGemmParams<T>* params) {
  return at::cuda::ScalarTypeToCudaDataType(params->bias_dtype);
}

template <typename T>
at::cuda::blas::GEMMAndBiasActivationEpilogue GetActivationFromParams(const GemmParams<T>* params) {
  return at::cuda::blas::GEMMAndBiasActivationEpilogue::None;
}

template <typename T>
at::cuda::blas::GEMMAndBiasActivationEpilogue GetActivationFromParams(const GemmAndBiasParams<T>* params) {
  return params->activation;
}

template <typename T>
at::cuda::blas::GEMMAndBiasActivationEpilogue GetActivationFromParams(const GemmStridedBatchedParams<T>* params) {
  return at::cuda::blas::GEMMAndBiasActivationEpilogue::None;
}

template <typename T>
at::cuda::blas::GEMMAndBiasActivationEpilogue GetActivationFromParams(const ScaledGemmParams<T>* params) {
  return at::cuda::blas::GEMMAndBiasActivationEpilogue::None;
}

static hipblasOperation_t _hipblasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return HIPBLAS_OP_N;
    case 't':
    case 'T':
      return HIPBLAS_OP_T;
    case 'c':
    case 'C':
      return HIPBLAS_OP_C;
  }
  TORCH_CHECK(false,
      "_hipblasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

static char _charFromhipblasOp(hipblasOperation_t op) {
  switch (op) {
    case HIPBLAS_OP_N:
      return 'N';
    case HIPBLAS_OP_T:
      return 'T';
    case HIPBLAS_OP_C:
      return 'C';
  }
  TORCH_CHECK(false,
      "_charFromhipblasOp input should be HIPBLAS_OP_N/T/C but got `", op, "`");
}

static hipblasOperation_t MapLayoutToHipBlasLt(BlasOp layout) {
  if (layout == BlasOp::N) {
    return HIPBLAS_OP_N;
  }
  return HIPBLAS_OP_T;
}

static size_t GetHipblasltWorkspaceSize() {
  static const auto env = c10::utils::get_env("HIPBLASLT_WORKSPACE_SIZE");
  // 256MB is max workspace size allowed for hipblaslt
  // hipblaslt-bench uses 32MB
  // recommendation from hipblaslt author was 76MB
  // TunableOp hipBLASLt workspace size is aligned with
  // PyTorch's default in CUDABlas.cpp (_parseChosenWorkspaceSize)
  size_t workspace_size = 76*1024;
  if (env) {
    try {
      workspace_size = std::stoi(env.value());
    } catch(std::invalid_argument const& e) {
      TORCH_WARN("invalid HIPBLASLT_WORKSPACE_SIZE,",
                 " using default workspace size of ", workspace_size, " KiB.");
    } catch(std::out_of_range const& e) {
      TORCH_WARN("HIPBLASLT_WORKSPACE_SIZE out of range,",
                 " using default workspace size of ", workspace_size, " KiB.");
    }
  }
  return workspace_size * 1024;
}

template <typename T, cublasStatus_t (*destructor)(T*)>
struct HipBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, hipblasStatus_t (*destructor)(T*)>
class HipBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, HipBlasLtDeleter<T, destructor>> descriptor_;
};

class HipBlasLtMatmulDescriptor : public HipBlasLtDescriptor<
                                     hipblasLtMatmulDescOpaque_t,
                                     &hipblasLtMatmulDescDestroy> {
 public:
  HipBlasLtMatmulDescriptor(
      hipblasComputeType_t compute_type,
      hipDataType scale_type) {
    hipblasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_HIPBLASLT_CHECK(
        hipblasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(hipblasLtMatmulDescAttributes_t attr, const T value) {
    TORCH_HIPBLASLT_CHECK(::hipblasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
class HipblasltGemmOp : public Callable<ParamsT> {
  public:
    HipblasltGemmOp(hipblasLtMatmulAlgo_t algo) : algo_{algo} {}

    TuningStatus Call(const ParamsT* params) override {
      hipblasOperation_t transa_outer = MapLayoutToHipBlasLt(ALayout);
      hipblasOperation_t transb_outer = MapLayoutToHipBlasLt(BLayout);
      auto a_datatype = HipDataTypeFor<AT>();
      auto b_datatype = HipDataTypeFor<BT>();
      auto in_out_datatype = HipDataTypeFor<CT>();
      auto opa = _hipblasOpFromChar(params->transa);
      auto opb = _hipblasOpFromChar(params->transb);

      TORCH_CHECK(transa_outer == opa && transb_outer == opb, "trans mismatch, shouldn't happen");

      float alpha = GetAlphaFromParams<CT>(params);
      float beta = GetBetaFromParams<CT>(params);

      hipblasLtMatrixLayout_t mat_a, mat_b, mat_c;
      if (opa == HIPBLAS_OP_N) {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, a_datatype, params->m, params->k, params->lda));
      }
      else {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, a_datatype, params->k, params->m, params->lda));
      }
      if (opb == HIPBLAS_OP_N) {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, b_datatype, params->k, params->n, params->ldb));
      }
      else {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, b_datatype, params->n, params->k, params->ldb));
      }
      TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, in_out_datatype, params->m, params->n, params->ldc));

      // specific to batched gemmm
      int batch = GetBatchFromParams<CT>(params);
      if (batch > 1) {
        int64_t stride_a = GetStrideAFromParams<CT>(params);
        int64_t stride_b = GetStrideBFromParams<CT>(params);
        int64_t stride_c = GetStrideCFromParams<CT>(params);
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
            mat_a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
            mat_a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_a, sizeof(stride_a)));
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
            mat_b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
            mat_b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_b, sizeof(stride_b)));
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
            mat_c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutSetAttribute(
            mat_c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride_c, sizeof(stride_c)));
      }

      hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;
      if (at::globalContext().allowTF32CuBLAS()) {
        computeType = HIPBLAS_COMPUTE_32F_FAST_TF32;
      }
      HipBlasLtMatmulDescriptor matmul(computeType, HIP_R_32F);
      matmul.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSA, opa);
      matmul.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSB, opb);

      // specific to scaled gemm
      const void* mat1_scale_ptr = GetAScalePointerFromParams<CT>(params);
      const void* mat2_scale_ptr = GetBScalePointerFromParams<CT>(params);
      const void* result_scale_ptr = GetDScalePointerFromParams<CT>(params);
      if (mat1_scale_ptr && mat2_scale_ptr) {
#ifdef HIPBLASLT_VEC_EXT
        if (GetUseRowwiseFromParams<CT>(params)) {
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT, mat1_scale_ptr);
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT, mat2_scale_ptr);
        }
        else
#endif
        {
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, mat1_scale_ptr);
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, mat2_scale_ptr);
        }
      }
      if (result_scale_ptr) {
        matmul.setAttribute(HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, result_scale_ptr);
      }

      const void* bias_ptr = GetBiasPointerFromParams<CT>(params);
      auto bias_datatype = GetBiasTypeFromParams<CT>(params);
      if (bias_ptr) {
        matmul.setAttribute(HIPBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
        matmul.setAttribute(HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_datatype);
        auto activation = GetActivationFromParams<CT>(params);
        if (activation == at::cuda::blas::GEMMAndBiasActivationEpilogue::RELU) {
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_EPILOGUE, HIPBLASLT_EPILOGUE_RELU_BIAS);
        }
        else if (activation == at::cuda::blas::GEMMAndBiasActivationEpilogue::GELU) {
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_EPILOGUE, HIPBLASLT_EPILOGUE_GELU_BIAS);
        }
        else {
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_EPILOGUE, HIPBLASLT_EPILOGUE_BIAS);
        }
      }

      size_t workspace_size = GetHipblasltWorkspaceSize();

      auto op_handle = at::cuda::getCurrentCUDABlasLtHandle();

      size_t ret_workspace_size = 0;
      auto status = hipblaslt_ext::matmulIsAlgoSupported(op_handle,
          matmul.descriptor(),
          &alpha,
          mat_a,
          mat_b,
          &beta,
          mat_c,
          mat_c,
          algo_,
          ret_workspace_size);

      if (status == HIPBLAS_STATUS_SUCCESS) {
        if (ret_workspace_size >= workspace_size) {
          return FAIL;
        }
      }
      else {
        return FAIL;
      }

      void* workspace_buffer = nullptr;
      if (workspace_size > 0) {
        workspace_buffer = c10::cuda::CUDACachingAllocator::raw_alloc(workspace_size);
      }

      TORCH_HIPBLASLT_CHECK(hipblasLtMatmul(op_handle,
            matmul.descriptor(),
            &alpha,
            params->a,
            mat_a,
            params->b,
            mat_b,
            &beta,
            params->c,
            mat_c,
            params->c,
            mat_c,
            &algo_,
            workspace_buffer,
            workspace_size,
            at::cuda::getCurrentCUDAStream()));

      //TORCH_HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
      TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_a));
      TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_b));
      TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(mat_c));
      if (workspace_size > 0) {
        c10::cuda::CUDACachingAllocator::raw_delete(workspace_buffer);
      }
      return OK;
    }

  private:
    hipblasLtMatmulAlgo_t algo_;
};

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
auto GetHipBlasLtTypeStringAndOps() {
  hipblasOperation_t transa_outer = MapLayoutToHipBlasLt(ALayout);
  hipblasOperation_t transb_outer = MapLayoutToHipBlasLt(BLayout);
  auto a_datatype = HipDataTypeFor<AT>();
  auto b_datatype = HipDataTypeFor<BT>();
  auto in_out_datatype = HipDataTypeFor<CT>();
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;

  hipblasComputeType_t computeType = HIPBLAS_COMPUTE_32F;
  if (at::globalContext().allowTF32CuBLAS()) {
    computeType = HIPBLAS_COMPUTE_32F_FAST_TF32;
  }

  hipblasLtHandle_t handle;
  TORCH_HIPBLASLT_CHECK(hipblasLtCreate(&handle));
  TORCH_HIPBLASLT_CHECK(hipblaslt_ext::getAllAlgos(handle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        transa_outer,
        transb_outer,
        a_datatype,
        b_datatype,
        in_out_datatype,
        in_out_datatype,
        computeType,
        heuristic_result));
  TORCH_HIPBLASLT_CHECK(hipblasLtDestroy(handle));

  int returned_algo_count = heuristic_result.size();
  std::vector<std::pair<std::string, std::unique_ptr<Callable<ParamsT>>>> ret;
  for (int i = 0; i < returned_algo_count; i++) {
    auto algo = heuristic_result[i].algo;
    int algo_index = hipblaslt_ext::getIndexFromAlgo(algo);
    auto callable = std::make_unique<HipblasltGemmOp<AT, BT, CT, ALayout, BLayout, ParamsT>>(algo);
    std::string type_string = fmt::sprintf("Gemm_Hipblaslt_%d", algo_index);
    ret.emplace_back(type_string, std::move(callable));
  }

  return ret;
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, T, T, ALayout, BLayout, GemmParams<T>>();
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmAndBiasTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, T, T, ALayout, BLayout, GemmAndBiasParams<T>>();
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmStridedBatchedTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, T, T, ALayout, BLayout, GemmStridedBatchedParams<T>>();
}

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtScaledGemmTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<AT, BT, CT, ALayout, BLayout, ScaledGemmParams<CT>>();
}

#undef TORCH_HIPBLASLT_CHECK

}  // namespace at::cuda::tunable
