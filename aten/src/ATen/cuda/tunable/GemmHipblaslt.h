// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
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

template <typename T, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
class HipblasltGemmOp : public Callable<ParamsT> {
  public:
    HipblasltGemmOp(hipblasLtMatmulAlgo_t algo) : algo_{algo} {}

    TuningStatus Call(const ParamsT* params) override {
      hipblasOperation_t transa_outer = MapLayoutToHipBlasLt(ALayout);
      hipblasOperation_t transb_outer = MapLayoutToHipBlasLt(BLayout);
      auto in_out_datatype = HipBlasDataTypeFor<T>();
      auto opa = _hipblasOpFromChar(params->transa);
      auto opb = _hipblasOpFromChar(params->transb);

      TORCH_CHECK(transa_outer == opa && transb_outer == opb, "trans mismatch, shouldn't happen");

      float alpha = static_cast<float>(params->alpha);
      float beta = static_cast<float>(params->beta);

      hipblasLtMatrixLayout_t mat_a, mat_b, mat_c;
      hipblasLtMatmulDesc_t matmul;
      if (opa == HIPBLAS_OP_N) {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, in_out_datatype, params->m, params->k, params->lda));
      }
      else {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_a, in_out_datatype, params->k, params->m, params->lda));
      }
      if (opb == HIPBLAS_OP_N) {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, in_out_datatype, params->k, params->n, params->ldb));
      }
      else {
        TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_b, in_out_datatype, params->n, params->k, params->ldb));
      }
      TORCH_HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&mat_c, in_out_datatype, params->m, params->n, params->ldc));
      TORCH_HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmul, COMPUTE_TYPE_32, DATA_TYPE_R_32));

      int batch = GetBatchFromParams<T>(params);
      if (batch > 1) {
        int64_t stride_a = GetStrideAFromParams<T>(params);
        int64_t stride_b = GetStrideBFromParams<T>(params);
        int64_t stride_c = GetStrideCFromParams<T>(params);
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

      TORCH_HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(int32_t)));
      TORCH_HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(int32_t)));

      size_t workspace_size = GetHipblasltWorkspaceSize();

      auto op_handle = at::cuda::getCurrentCUDABlasLtHandle();

      size_t ret_workspace_size = 0;
      auto status = hipblaslt_ext::matmulIsAlgoSupported(op_handle,
          matmul,
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
            matmul,
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

      TORCH_HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(matmul));
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

template <typename T, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
auto GetHipBlasLtTypeStringAndOps() {
  hipblasOperation_t transa_outer = MapLayoutToHipBlasLt(ALayout);
  hipblasOperation_t transb_outer = MapLayoutToHipBlasLt(BLayout);
  auto in_out_datatype = HipBlasDataTypeFor<T>();
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;

  hipblasLtHandle_t handle;
  TORCH_HIPBLASLT_CHECK(hipblasLtCreate(&handle));
  TORCH_HIPBLASLT_CHECK(hipblaslt_ext::getAllAlgos(handle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        transa_outer,
        transb_outer,
        in_out_datatype,
        in_out_datatype,
        in_out_datatype,
        in_out_datatype,
        COMPUTE_TYPE_32,
        heuristic_result));
  TORCH_HIPBLASLT_CHECK(hipblasLtDestroy(handle));

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
    auto callable = std::make_unique<HipblasltGemmOp<T, ALayout, BLayout, ParamsT>>(algo);
    std::string type_string = c10::str(
        "Gemm_Hipblaslt_", _charFromhipblasOp(transa_outer), _charFromhipblasOp(transb_outer), "_", algo_index);
    ret.emplace_back(type_string, std::move(callable));
  }

  return ret;
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, ALayout, BLayout, GemmParams<T>>();
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmStridedBatchedTypeStringAndOps() {
  return GetHipBlasLtTypeStringAndOps<T, ALayout, BLayout, GemmStridedBatchedParams<T>>();
}

#undef TORCH_HIPBLASLT_CHECK
#undef GETINDEXFROMALGO
#undef COMPUTE_TYPE_32
#undef DATA_TYPE_R_32

}  // namespace at::cuda::tunable
