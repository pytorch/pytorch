#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/StringUtil.h>

namespace at::cuda::tunable {

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
at::opmath_type<T> GetAlphaFromParams(const GemmParams<T>* params) {
    return params->alpha;
}

template <typename T>
at::opmath_type<T> GetAlphaFromParams(const GemmAndBiasParams<T>* params) {
    return params->alpha;
}

template <typename T>
at::opmath_type<T> GetAlphaFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->alpha;
}

template <typename T>
at::opmath_type<T> GetAlphaFromParams(const ScaledGemmParams<T>* params) {
  return 1.0;
}

template <typename T>
at::opmath_type<T> GetBetaFromParams(const GemmParams<T>* params) {
    return params->beta;
}

template <typename T>
at::opmath_type<T> GetBetaFromParams(const GemmAndBiasParams<T>* params) {
    return 0.0;
}

template <typename T>
at::opmath_type<T> GetBetaFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->beta;
}

template <typename T>
at::opmath_type<T> GetBetaFromParams(const ScaledGemmParams<T>* params) {
  return 0.0;
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
const void* GetAmaxPointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetAmaxPointerFromParams(const GemmAndBiasParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetAmaxPointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

template <typename T>
const void* GetAmaxPointerFromParams(const ScaledGemmParams<T>* params) {
  return params->amax_ptr;
}

template <typename T>
int8_t GetFastAccumModeFromParams(const GemmParams<T>* params) {
  return 0;
}

template <typename T>
int8_t GetFastAccumModeFromParams(const GemmAndBiasParams<T>* params) {
  return 0;
}

template <typename T>
int8_t GetFastAccumModeFromParams(const GemmStridedBatchedParams<T>* params) {
  return 0;
}

template <typename T>
int8_t GetFastAccumModeFromParams(const ScaledGemmParams<T>* params) {
  return params->use_fast_accum ? 1 : 0;
}

static cublasOperation_t MapLayoutToCuBlasLt(BlasOp layout) {
    if (layout == BlasOp::N) {
        return CUBLAS_OP_N;
    }
    return CUBLAS_OP_T;
}

template <typename T>
const void* GetBiasPointerFromParams(const GemmParams<T>*params) {
    return nullptr;
}

template <typename T>
const void* GetBiasPointerFromParams(const GemmAndBiasParams<T>*params) {
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
cudaDataType_t GetBiasTypeFromParams(const GemmParams<T>* params) {
    return CUDA_R_32F;
}

template <typename T>
cudaDataType_t GetBiasTypeFromParams(const GemmAndBiasParams<T>* params) {
    if (std::is_same_v<T, double>) {
      return CUDA_R_64F;
    } else if (std::is_same_v<T, float>) {
      return CUDA_R_32F;
    } else if (std::is_same_v<T, Half>) {
      return CUDA_R_16F;
    } else if (std::is_same_v<T, BFloat16>) {
      return CUDA_R_16BF;
    }
}

template <typename T>
cudaDataType_t GetBiasTypeFromParams(const GemmStridedBatchedParams<T>* params) {
    return CUDA_R_32F;
}

template <typename T>
cudaDataType_t GetBiasTypeFromParams(const ScaledGemmParams<T>* params) {
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

static char _charFromcublasOp(cublasOperation_t op) {
    switch (op) {
        case CUBLAS_OP_N:
            return 'N';
        case CUBLAS_OP_T:
            return 'T';
        case CUBLAS_OP_C:
            return 'C';
        default:
        AT_ERROR("_charFromcublasOp input should be CUBLAS_OP_N/T/C but got '", op, "'");
    }
}

template <typename T, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
class CublasltGemmOp : public Callable<ParamsT> {
    public:
        CublasltGemmOp(cublasLtMatmulAlgo_t algo) : algo_{algo} {}

        TuningStatus Call(const ParamsT* params) override {
            // Follows `bgemm_internal_cublaslt` and `scaled_gemm` closely, should be kept in sync with the non-tunable code path.
            at::opmath_type<T> alpha = GetAlphaFromParams<T>(params);
            at::opmath_type<T> beta = GetBetaFromParams<T>(params);

            cudaDataType_t a_dtype, b_dtype, c_dtype;
            cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
            cudaDataType_t scaleType = CUDA_R_32F;

            if constexpr (std::is_same_v<ParamsT, GemmParams<T>> || std::is_same_v<ParamsT, GemmAndBiasParams<T>> || std::is_same_v<ParamsT, GemmStridedBatchedParams<T>>) {
                cudaDataType_t abcType = CUDA_R_32F;
                if constexpr (std::is_same_v<T, double>) {
                    abcType = CUDA_R_64F;
                    computeType = CUBLAS_COMPUTE_64F;
                    scaleType = CUDA_R_64F;
                } else if constexpr (std::is_same_v<T, float>) {
                    if (at::globalContext().allowTF32CuBLAS()) {
                        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
                    }
                } else if constexpr (std::is_same_v<T, c10::complex<double>>) {
                    abcType = CUDA_C_64F;
                    computeType = CUBLAS_COMPUTE_64F;
                    scaleType = CUDA_C_64F;
                } else if constexpr (std::is_same_v<T, c10::complex<float>>) {
                    abcType = CUDA_C_32F;
                    scaleType = CUDA_C_32F;
                } else if constexpr (std::is_same_v<T, at::Half>) {
                    abcType = CUDA_R_16F;
                } else if constexpr (std::is_same_v<T, at::BFloat16>) {
                    abcType = CUDA_R_16BF;
                } else {
                    static_assert(false && sizeof(T), "at::cuda::tunable::CublasltGemmOp: not implemented");
                }

                a_dtype = abcType;
                b_dtype = abcType;
                c_dtype = abcType;
            } else if constexpr (std::is_same_v<ParamsT, ScaledGemmParams<T>>) {
                a_dtype = ScalarTypeToCudaDataType(((ScaledGemmParams<T>*)params)->a_dtype);
                b_dtype = ScalarTypeToCudaDataType(((ScaledGemmParams<T>*)params)->b_dtype);
                c_dtype = ScalarTypeToCudaDataType(((ScaledGemmParams<T>*)params)->c_dtype);
            }

            globalContext().alertCuBLASConfigNotDeterministic();
            cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

            cublasOperation_t transa_outer = MapLayoutToCuBlasLt(ALayout);
            cublasOperation_t transb_outer = MapLayoutToCuBlasLt(BLayout);
            cublasOperation_t opa = at::cuda::blas::_cublasOpFromChar(params->transa);
            cublasOperation_t opb = at::cuda::blas::_cublasOpFromChar(params->transb);
            TORCH_CHECK(transa_outer == opa && transb_outer == opb, "trans mismatch, shouldn't happen");

            int64_t lda, ldb, ldc;
            lda = params->lda;
            ldb = params->ldb;
            ldc = params->ldc;
            at::cuda::blas::_cublasAdjustLdLevel3(params->transa, params->transb, params->m, params->n, params->k, &lda, &ldb, &ldc);

            at::cuda::blas::CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
            computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
            computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);

            const void* mat1_scale_ptr = GetAScalePointerFromParams<T>(params);
            const void* mat2_scale_ptr = GetBScalePointerFromParams<T>(params);
            const void* mat3_scale_ptr = GetDScalePointerFromParams<T>(params);
            const void* amax_ptr = GetAmaxPointerFromParams<T>(params);
            const int8_t fast_accum_mode = GetFastAccumModeFromParams<T>(params);
            if (mat1_scale_ptr && mat2_scale_ptr) {
                computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, mat1_scale_ptr);
                computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, mat2_scale_ptr);

                if (amax_ptr) {
                    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, amax_ptr);
                }

                computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum_mode);
            }

            if (mat3_scale_ptr) {
                computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, mat3_scale_ptr);
            }

            const void* bias_ptr = GetBiasPointerFromParams<T>(params);
            auto bias_datatype = GetBiasTypeFromParams<T>(params);
            if (bias_ptr) {
                computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
                computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_datatype);

                auto activation = GetActivationFromParams<T>(params);
                if (activation == at::cuda::blas::GEMMAndBiasActivationEpilogue::RELU) {
                    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_RELU_BIAS);
                } else if (activation == at::cuda::blas::GEMMAndBiasActivationEpilogue::GELU) {
                    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_GELU_BIAS);
                } else {
                    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
                }
            }

            at::cuda::blas::CuBlasLtMatrixLayout Adesc(a_dtype, params->m, params->k, lda, opa == CUBLAS_OP_T);
            at::cuda::blas::CuBlasLtMatrixLayout Bdesc(b_dtype, params->k, params->n, ldb, opb == CUBLAS_OP_T);
            at::cuda::blas::CuBlasLtMatrixLayout Cdesc(c_dtype, params->m, params->n, ldc);
            if constexpr (std::is_same_v<ParamsT, ScaledGemmParams<T>>) {
                Cdesc = at::cuda::blas::CuBlasLtMatrixLayout(bias_datatype, params->m, params->n, ldc);
            }
            at::cuda::blas::CuBlasLtMatrixLayout Ddesc(c_dtype, params->m, params->n, ldc);

            int batch_size = GetBatchFromParams<T>(params);
            if (batch_size > 1) {
                int64_t stride_a = GetStrideAFromParams<T>(params);
                int64_t stride_b = GetStrideBFromParams<T>(params);
                int64_t stride_c = GetStrideCFromParams<T>(params);
                Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
                Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
                Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
                Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);
                Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b);
                Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_c);
            }

            at::cuda::blas::CuBlasLtMatmulPreference preference;
            // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
            // setting this to 1M.
            size_t workspaceSize = at::cuda::blas::_getWorkspaceSize();
            preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

            uint32_t a_alignment = at::cuda::blas::_getAlignment(reinterpret_cast<uintptr_t>(params->a));
            uint32_t b_alignment = at::cuda::blas::_getAlignment(reinterpret_cast<uintptr_t>(params->b));
            uint32_t c_alignment = at::cuda::blas::_getAlignment(reinterpret_cast<uintptr_t>(params->c));
            preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
            preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
            preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);

            auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
            auto workspace = allocator.allocate(workspaceSize);
            TORCH_CHECK(workspace.get() != nullptr, "OOM trying to allocate workspace for cublaslt");

            cublasLtMatmulHeuristicResult_t heuristic_results;
            auto status = cublasLtMatmulAlgoCheck(
                ltHandle,
                computeDesc.descriptor(),
                Adesc.descriptor(),
                Bdesc.descriptor(),
                Cdesc.descriptor(),
                std::is_same_v<ParamsT, ScaledGemmParams<T>> ? Ddesc.descriptor() : Cdesc.descriptor(),
                &algo_,
                &heuristic_results
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                return FAIL;
            }

            cublasStatus_t cublasStatus = cublasLtMatmul(
                ltHandle,
                computeDesc.descriptor(),
                &alpha,
                params->a,
                Adesc.descriptor(),
                params->b,
                Bdesc.descriptor(),
                &beta,
                params->c,
                Cdesc.descriptor(),
                params->c,
                std::is_same_v<ParamsT, ScaledGemmParams<T>> ? Ddesc.descriptor() : Cdesc.descriptor(),
                &algo_,
                workspace.mutable_get(),
                workspaceSize,
                at::cuda::getCurrentCUDAStream());
            TORCH_CHECK(
                cublasStatus == CUBLAS_STATUS_SUCCESS,
                "CUDA error: ",
                at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
                " when calling cublasLtMatmul with transpose_mat1 ",
                (opa == CUBLAS_OP_T),
                " transpose_mat2 ",
                (opb == CUBLAS_OP_T),
                " m ",
                params->m,
                " n ",
                params->n,
                " k ",
                params->k,
                " lda ",
                params->lda,
                " ldb ",
                params->ldb,
                " ldc ",
                params->ldc,
                " aType ",
                a_dtype,
                " bType ",
                b_dtype,
                " cType ",
                c_dtype,
                " dType ",
                c_dtype,
                " computeType ",
                computeType,
                " scaleType ",
                scaleType);

          return OK;
      }
    private:
        cublasLtMatmulAlgo_t algo_;
};

template <typename T, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
auto GetCublasLtTypeStringAndOps(const ParamsT* params) {
    cudaDataType_t a_dtype, b_dtype, c_dtype;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;

    if constexpr (std::is_same_v<ParamsT, GemmParams<T>> || std::is_same_v<ParamsT, GemmAndBiasParams<T>> || std::is_same_v<ParamsT, GemmStridedBatchedParams<T>>) {
        cudaDataType_t abcType = CUDA_R_32F;
        if constexpr (std::is_same_v<T, double>) {
            abcType = CUDA_R_64F;
            computeType = CUBLAS_COMPUTE_64F;
            scaleType = CUDA_R_64F;
        } else if constexpr (std::is_same_v<T, float>) {
            if (at::globalContext().allowTF32CuBLAS()) {
                computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
            }
        } else if constexpr (std::is_same_v<T, c10::complex<double>>) {
            abcType = CUDA_C_64F;
            computeType = CUBLAS_COMPUTE_64F;
            scaleType = CUDA_C_64F;
        } else if constexpr (std::is_same_v<T, c10::complex<float>>) {
            abcType = CUDA_C_32F;
            scaleType = CUDA_C_32F;
        } else if constexpr (std::is_same_v<T, at::Half>) {
            abcType = CUDA_R_16F;
        } else if constexpr (std::is_same_v<T, at::BFloat16>) {
            abcType = CUDA_R_16BF;
        } else {
            static_assert(false && sizeof(T), "at::cuda::tunable::CublasltGemmOp: not implemented");
        }

        a_dtype = abcType;
        b_dtype = abcType;
        c_dtype = abcType;
    } else if constexpr (std::is_same_v<ParamsT, ScaledGemmParams<T>>) {
        a_dtype = ScalarTypeToCudaDataType(((ScaledGemmParams<T>*)params)->a_dtype);
        b_dtype = ScalarTypeToCudaDataType(((ScaledGemmParams<T>*)params)->b_dtype);
        c_dtype = ScalarTypeToCudaDataType(((ScaledGemmParams<T>*)params)->c_dtype);
    }

    globalContext().alertCuBLASConfigNotDeterministic();
    cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

    cublasOperation_t transa_outer = MapLayoutToCuBlasLt(ALayout);
    cublasOperation_t transb_outer = MapLayoutToCuBlasLt(BLayout);
    cublasOperation_t opa = at::cuda::blas::_cublasOpFromChar(params->transa);
    cublasOperation_t opb = at::cuda::blas::_cublasOpFromChar(params->transb);
    TORCH_CHECK(transa_outer == opa && transb_outer == opb, "trans mismatch, shouldn't happen");

    int64_t lda, ldb, ldc;
    lda = params->lda;
    ldb = params->ldb;
    ldc = params->ldc;
    at::cuda::blas::_cublasAdjustLdLevel3(params->transa, params->transb, params->m, params->n, params->k, &lda, &ldb, &ldc);

    at::cuda::blas::CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);

    const void* mat1_scale_ptr = GetAScalePointerFromParams<T>(params);
    const void* mat2_scale_ptr = GetBScalePointerFromParams<T>(params);
    const void* mat3_scale_ptr = GetDScalePointerFromParams<T>(params);
    const void* amax_ptr = GetAmaxPointerFromParams<T>(params);
    const int8_t fast_accum_mode = GetFastAccumModeFromParams<T>(params);
    const void* bias_ptr = GetBiasPointerFromParams<T>(params);
    auto bias_datatype = GetBiasTypeFromParams<T>(params);
    if (mat1_scale_ptr && mat2_scale_ptr && mat3_scale_ptr) {
        computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, mat1_scale_ptr);
        computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, mat2_scale_ptr);
        computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, mat3_scale_ptr);

        if (amax_ptr) {
            computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, amax_ptr);
        }

        computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum_mode);
        if (bias_ptr) {
            computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
            computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
            computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_datatype);
        }
    }

    at::cuda::blas::CuBlasLtMatrixLayout Adesc(a_dtype, params->m, params->k, lda, opa == CUBLAS_OP_T);
    at::cuda::blas::CuBlasLtMatrixLayout Bdesc(b_dtype, params->k, params->n, ldb, opb == CUBLAS_OP_T);
    at::cuda::blas::CuBlasLtMatrixLayout Cdesc(c_dtype, params->m, params->n, ldc);
    if constexpr (std::is_same_v<ParamsT, ScaledGemmParams<T>>) {
        Cdesc = at::cuda::blas::CuBlasLtMatrixLayout(bias_datatype, params->m, params->n, ldc);
    }
    at::cuda::blas::CuBlasLtMatrixLayout Ddesc(c_dtype, params->m, params->n, ldc);

    int batch_size = GetBatchFromParams<T>(params);
    if (batch_size > 1) {
        int64_t stride_a = GetStrideAFromParams<T>(params);
        int64_t stride_b = GetStrideBFromParams<T>(params);
        int64_t stride_c = GetStrideCFromParams<T>(params);
        Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_size);
        Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a);
        Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b);
        Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_c);
    }

    at::cuda::blas::CuBlasLtMatmulPreference preference;
    // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
    // setting this to 1M.
    size_t workspaceSize = at::cuda::blas::_getWorkspaceSize();
    preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);

    uint32_t a_alignment = at::cuda::blas::_getAlignment(reinterpret_cast<uintptr_t>(params->a));
    uint32_t b_alignment = at::cuda::blas::_getAlignment(reinterpret_cast<uintptr_t>(params->b));
    uint32_t c_alignment = at::cuda::blas::_getAlignment(reinterpret_cast<uintptr_t>(params->c));
    preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
    preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
    preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);

    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    auto workspace = allocator.allocate(workspaceSize);
    TORCH_CHECK(workspace.get() != nullptr, "OOM trying to allocate workspace for cublaslt");

    cublasLtMatmulHeuristicResult_t heuristicResults[8];
    int returnedResults = 0;

    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle,
        computeDesc.descriptor(),
        Adesc.descriptor(),
        Bdesc.descriptor(),
        Cdesc.descriptor(),
        std::is_same_v<ParamsT, ScaledGemmParams<T>> ? Ddesc.descriptor() : Cdesc.descriptor(),
        preference.descriptor(),
        8,
        heuristicResults,
        &returnedResults));

    if (returnedResults == 0) {
        TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    std::vector<std::pair<std::string, std::unique_ptr<Callable<ParamsT>>>> ret;

    for (int i = 0; i < returnedResults; i++) {
        auto algo = heuristicResults[i].algo;
        auto callable = std::make_unique<CublasltGemmOp<T, ALayout, BLayout, ParamsT>>(algo);
        std::string type_string = c10::str("Gemm_Cublaslt_", _charFromcublasOp(transa_outer), _charFromcublasOp(transb_outer), "_", i);
        ret.emplace_back(type_string, std::move(callable));
    }

    return ret;
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetCublasLtGemmTypeStringAndOps(const GemmParams<T>* params) {
    return GetCublasLtTypeStringAndOps<T, ALayout, BLayout, GemmParams<T>>(params);
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetCublasLtGemmAndBiasTypeStringAndOps(const GemmAndBiasParams<T>* params) {
    return GetCublasLtTypeStringAndOps<T, ALayout, BLayout, GemmAndBiasParams<T>>(params);
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetCublasLtStridedBatchedGemmTypeStringAndOps(const GemmStridedBatchedParams<T>* params) {
    return GetCublasLtTypeStringAndOps<T, ALayout, BLayout, GemmStridedBatchedParams<T>>(params);
}
template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetCublasLtScaledGemmTypeStringAndOps(const ScaledGemmParams<T>* params) {
    return GetCublasLtTypeStringAndOps<T, ALayout, BLayout, ScaledGemmParams<T>>(params);
}

}  // namespace at::cuda::tunable
