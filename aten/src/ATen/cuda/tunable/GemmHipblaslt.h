// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/StringUtil.h>

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

#ifdef HIPBLASLT_HAS_GETINDEXFROMALGO
#define GETINDEXFROMALGO(algo) hipblaslt_ext::getIndexFromAlgo(algo)
#else
static int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo) {
    int* algo_ptr = (int*)algo.data;
    if(*algo_ptr < 0) {
        return -1;
    }
    return *algo_ptr;
}
#define GETINDEXFROMALGO(algo) getIndexFromAlgo(algo)
#endif

#ifdef HIPBLASLT_CUSTOM_COMPUTE_TYPE
#define COMPUTE_TYPE_32 HIPBLASLT_COMPUTE_F32
#else
#define COMPUTE_TYPE_32 HIPBLAS_COMPUTE_32F
#endif

#ifdef HIPBLASLT_CUSTOM_DATA_TYPE

template <typename T>
constexpr hipblasltDatatype_t HipBlasDataTypeFor();

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<float>() {
  return HIPBLASLT_R_32F;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<Half>() {
  return HIPBLASLT_R_16F;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<BFloat16>() {
  return HIPBLASLT_R_16B;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<double>() {
  return HIPBLASLT_R_64F;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<c10::Float8_e4m3fnuz>() {
  return HIPBLASLT_R_8F_E4M3;
}

template <>
constexpr hipblasltDatatype_t HipBlasDataTypeFor<c10::Float8_e5m2fnuz>() {
  return HIPBLASLT_R_8F_E5M3;
}

#define DATA_TYPE_R_32 HIPBLASLT_R_32F

#else

template <typename T>
constexpr hipblasDatatype_t HipBlasDataTypeFor();

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<float>() {
  return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<Half>() {
  return HIPBLAS_R_16F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<BFloat16>() {
  return HIPBLAS_R_16B;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<double>() {
  return HIPBLAS_R_64F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<c10::Float8_e4m3fnuz>() {
  return HIP_R_8F_E4M3_FNUZ;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<c10::Float8_e5m2fnuz>() {
  return HIP_R_8F_E5M2_FNUZ;
}

#ifdef HIPBLAS_V2
#define DATA_TYPE_R_32 HIP_R_32F
#else
#define DATA_TYPE_R_32 HIPBLAS_R_32F
#endif

#endif

template <typename T, typename ParamsT>
int GetBatchFromParams(const ParamsT* params) {
  return 1;
}

template <typename T>
int GetBatchFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->batch;
}

template <typename T, typename ParamsT>
int GetStrideAFromParams(const ParamsT* params) {
  return 1;
}

template <typename T>
int GetStrideAFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_a;
}

template <typename T, typename ParamsT>
int GetStrideBFromParams(const ParamsT* params) {
  return 1;
}

template <typename T>
int GetStrideBFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_b;
}

template <typename T, typename ParamsT>
int GetStrideCFromParams(const ParamsT* params) {
  return 1;
}

template <typename T>
int GetStrideCFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_c;
}

template <typename T, typename ParamsT>
float GetAlphaFromParams(const ParamsT* params) {
  return params->alpha;
}

template <typename T>
float GetAlphaFromParams(const ScaledGemmParams<T>* params) {
  return 1.0;
}

template <typename T, typename ParamsT>
float GetBetaFromParams(const ParamsT* params) {
  return params->beta;
}

template <typename T>
float GetBetaFromParams(const ScaledGemmParams<T>* params) {
  return 0.0;
}

template <typename T, typename ParamsT>
bool GetFastAccuModeFromParams(const ParamsT* params) {
  return false;
}

template <typename T>
bool GetFastAccuModeFromParams(const ScaledGemmParams<T>* params) {
  return params->use_fast_accum;
}

template <typename T, typename ParamsT>
const void* GetAScalePointerFromParams(const ParamsT* params) {
  return nullptr;
}

template <typename T>
const void* GetAScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->a_scale_ptr;
}

template <typename T, typename ParamsT>
const void* GetBScalePointerFromParams(const ParamsT* params) {
  return nullptr;
}

template <typename T>
const void* GetBScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->b_scale_ptr;
}

template <typename T, typename ParamsT>
const void* GetDScalePointerFromParams(const ParamsT* params) {
  return nullptr;
}

template <typename T>
const void* GetDScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->c_scale_ptr;
}

template <typename T, typename ParamsT>
const void* GetBiasPointerFromParams(const ParamsT* params) {
  return nullptr;
}

template <typename T>
const void* GetBiasPointerFromParams(const ScaledGemmParams<T>* params) {
  return params->bias_ptr;
}

template <typename T, typename ParamsT>
hipDataType GetBiasTypeFromParams(const ParamsT* params) {
  return HIP_R_32F;
}

template <typename T>
hipDataType GetBiasTypeFromParams(const ScaledGemmParams<T>* params) {
  return at::cuda::ScalarTypeToCudaDataType(params->bias_dtype);
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
  AT_ERROR(
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
  AT_ERROR(
      "_charFromhipblasOp input should be HIPBLAS_OP_N/T/C but got `", op, "`");
}

static hipblasOperation_t MapLayoutToHipBlasLt(BlasOp layout) {
  if (layout == BlasOp::N) {
    return HIPBLAS_OP_N;
  }
  return HIPBLAS_OP_T;
}

static size_t GetHipblasltWorkspaceSize() {
  static const char * env = getenv("HIPBLASLT_WORKSPACE_SIZE");
  // 256MB is max workspace size allowed for hipblaslt
  // hipblaslt-bench uses 32MB
  // recommendation from hipblaslt author was 76MB
  size_t workspace_size = 2*128*1024*1024; // default 256MB
  if (env) {
    try {
      workspace_size = std::stoi(env);
    } catch(std::invalid_argument const& e) {
      TORCH_WARN("invalid HIPBLASLT_WORKSPACE_SIZE,",
                 " using default workspace size of ", workspace_size, " bytes.");
    } catch(std::out_of_range const& e) {
      TORCH_WARN("HIPBLASLT_WORKSPACE_SIZE out of range,",
                 " using default workspace size of ", workspace_size, " bytes.");
    }
  }
  return workspace_size;
}

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
class HipblasltGemmOp : public Callable<ParamsT> {
  public:
    HipblasltGemmOp(hipblasLtMatmulAlgo_t algo) : algo_{algo} {}

    TuningStatus Call(const ParamsT* params) override {
      hipblasOperation_t transa_outer = MapLayoutToHipBlasLt(ALayout);
      hipblasOperation_t transb_outer = MapLayoutToHipBlasLt(BLayout);
      auto a_datatype = HipBlasDataTypeFor<AT>();
      auto b_datatype = HipBlasDataTypeFor<BT>();
      auto in_out_datatype = HipBlasDataTypeFor<CT>();
      auto opa = _hipblasOpFromChar(params->transa);
      auto opb = _hipblasOpFromChar(params->transb);

      TORCH_CHECK(transa_outer == opa && transb_outer == opb, "trans mismatch, shouldn't happen");

      float alpha = GetAlphaFromParams<CT>(params);
      float beta = GetBetaFromParams<CT>(params);
      //bool use_fast_accum = GetFastAccuModeFromParams<CT>(params);

      at::cuda::blas::CuBlasLtMatrixLayout mat_a(a_datatype, params->m, params->k, params->lda, opa == HIPBLAS_OP_N);
      at::cuda::blas::CuBlasLtMatrixLayout mat_b(b_datatype, params->k, params->n, params->ldb, opb == HIPBLAS_OP_N);
      at::cuda::blas::CuBlasLtMatrixLayout mat_c(in_out_datatype, params->m, params->n, params->ldc);

      // specific to batched gemmm
      int batch = GetBatchFromParams<CT>(params);
      if (batch > 1) {
        int64_t stride_a = GetStrideAFromParams<CT>(params);
        int64_t stride_b = GetStrideBFromParams<CT>(params);
        int64_t stride_c = GetStrideCFromParams<CT>(params);
        mat_a.setAttribute(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
        mat_b.setAttribute(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
        mat_c.setAttribute(HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch);
        mat_a.setAttribute(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);
        mat_b.setAttribute(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b);
        mat_c.setAttribute(HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_c);
      }

      at::cuda::blas::CuBlasLtMatmulDescriptor matmul(COMPUTE_TYPE_32, DATA_TYPE_R_32);
      matmul.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSA, opa);
      matmul.setAttribute(HIPBLASLT_MATMUL_DESC_TRANSB, opb);

      // specific to scaled gemm
      const void* mat1_scale_ptr = GetAScalePointerFromParams<CT>(params);
      const void* mat2_scale_ptr = GetBScalePointerFromParams<CT>(params);
      const void* result_scale_ptr = GetDScalePointerFromParams<CT>(params);
      if (mat1_scale_ptr && mat2_scale_ptr && result_scale_ptr) {
        matmul.setAttribute(HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, mat1_scale_ptr);
        matmul.setAttribute(HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, mat2_scale_ptr);
        matmul.setAttribute(HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, result_scale_ptr);

        const void* bias_ptr = GetBiasPointerFromParams<CT>(params);
        auto bias_datatype = GetBiasTypeFromParams<CT>(params);
        if (bias_ptr) {
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_EPILOGUE, HIPBLASLT_EPILOGUE_BIAS);
          matmul.setAttribute(HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_datatype);
        }
      }

      size_t workspace_size = GetHipblasltWorkspaceSize();

      auto op_handle = at::cuda::getCurrentCUDABlasLtHandle();

      size_t ret_workspace_size = 0;
      auto status = hipblaslt_ext::matmulIsAlgoSupported(op_handle,
          matmul.descriptor(),
          &alpha,
          mat_a.descriptor(),
          mat_b.descriptor(),
          &beta,
          mat_c.descriptor(),
          mat_c.descriptor(),
          algo_,
          ret_workspace_size);

      if (status == HIPBLAS_STATUS_SUCCESS) {
        if (ret_workspace_size >= workspace_size) {
          //TUNABLE_LOG("[hipBLASLt] Solution #", algo_index, " workspace too large");
          return FAIL;
        }
      }
      else {
        //TUNABLE_LOG("[hipBLASLt] Solution #", algo_index, " not supported");
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
            mat_a.descriptor(),
            params->b,
            mat_b.descriptor(),
            &beta,
            params->c,
            mat_c.descriptor(),
            params->c,
            mat_c.descriptor(),
            &algo_,
            workspace_buffer,
            workspace_size,
            at::cuda::getCurrentCUDAStream()));

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
  auto a_datatype = HipBlasDataTypeFor<AT>();
  auto b_datatype = HipBlasDataTypeFor<BT>();
  auto in_out_datatype = HipBlasDataTypeFor<CT>();
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;

  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_HIPBLASLT_CHECK(hipblaslt_ext::getAllAlgos(handle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        transa_outer,
        transb_outer,
        a_datatype,
        b_datatype,
        in_out_datatype,
        in_out_datatype,
        COMPUTE_TYPE_32,
        heuristic_result));

  // Sort heuristic_result by algo index to make sure the order of returned algos is deterministic.
  std::sort(heuristic_result.begin(),
      heuristic_result.end(),
      [](hipblasLtMatmulHeuristicResult_t& a, hipblasLtMatmulHeuristicResult_t& b) {
      return GETINDEXFROMALGO(a.algo) < GETINDEXFROMALGO(b.algo);
      });

  int returned_algo_count = heuristic_result.size();
  std::vector<std::pair<std::string, std::unique_ptr<Callable<ParamsT>>>> ret;
  for (int i = 0; i < returned_algo_count; i++) {
    auto algo = heuristic_result[i].algo;
    int algo_index = GETINDEXFROMALGO(algo);
    auto callable = std::make_unique<HipblasltGemmOp<AT, BT, CT, ALayout, BLayout, ParamsT>>(algo);
    std::string type_string = c10::str(
        "Gemm_Hipblaslt_", _charFromhipblasOp(transa_outer), _charFromhipblasOp(transb_outer), "_", algo_index);
    ret.emplace_back(type_string, std::move(callable));
  }

  return ret;
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, T, T, ALayout, BLayout, GemmParams<T>>();
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
#undef GETINDEXFROMALGO
#undef COMPUTE_TYPE_32
#undef DATA_TYPE_R_32

}  // namespace at::cuda::tunable
