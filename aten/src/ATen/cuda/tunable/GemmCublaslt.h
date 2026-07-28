#pragma once

#ifndef USE_ROCM

#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/detail/BLASConstants.h>
#include <ATen/cuda/detail/CublasLtUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <fmt/printf.h>

#include <cstdio>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::cuda::tunable {

namespace {

template <typename T>
int64_t CublasltBatchCount(const GemmParams<T>* /*params*/) {
  return 0;
}

template <typename T>
int64_t CublasltBatchCount(const GemmAndBiasParams<T>* /*params*/) {
  return 0;
}

template <typename T, typename C_Dtype>
int64_t CublasltBatchCount(
    const GemmStridedBatchedParams<T, C_Dtype>* params) {
  return params->batch;
}

template <typename T>
int64_t CublasltStrideA(const GemmParams<T>* /*params*/) {
  return 0;
}

template <typename T>
int64_t CublasltStrideA(const GemmAndBiasParams<T>* /*params*/) {
  return 0;
}

template <typename T, typename C_Dtype>
int64_t CublasltStrideA(const GemmStridedBatchedParams<T, C_Dtype>* params) {
  return params->stride_a;
}

template <typename T>
int64_t CublasltStrideB(const GemmParams<T>* /*params*/) {
  return 0;
}

template <typename T>
int64_t CublasltStrideB(const GemmAndBiasParams<T>* /*params*/) {
  return 0;
}

template <typename T, typename C_Dtype>
int64_t CublasltStrideB(const GemmStridedBatchedParams<T, C_Dtype>* params) {
  return params->stride_b;
}

template <typename T>
int64_t CublasltStrideC(const GemmParams<T>* /*params*/) {
  return 0;
}

template <typename T>
int64_t CublasltStrideC(const GemmAndBiasParams<T>* /*params*/) {
  return 0;
}

template <typename T, typename C_Dtype>
int64_t CublasltStrideC(const GemmStridedBatchedParams<T, C_Dtype>* params) {
  return params->stride_c;
}

template <typename T>
at::opmath_type<T> CublasltBeta(const GemmParams<T>* params) {
  return params->beta;
}

template <typename T, typename C_Dtype>
at::opmath_type<T> CublasltBeta(
    const GemmStridedBatchedParams<T, C_Dtype>* params) {
  return params->beta;
}

template <typename T>
at::opmath_type<T> CublasltBeta(const GemmAndBiasParams<T>* params) {
  return params->bias ? 0 : 1;
}

template <typename ParamsT>
const void* CublasltBias(const ParamsT* /*params*/) {
  return nullptr;
}

template <typename T>
const void* CublasltBias(const GemmAndBiasParams<T>* params) {
  return params->bias;
}

template <typename ParamsT>
at::cuda::blas::GEMMAndBiasActivationEpilogue CublasltActivation(
    const ParamsT* /*params*/) {
  return at::cuda::blas::GEMMAndBiasActivationEpilogue::None;
}

template <typename T>
at::cuda::blas::GEMMAndBiasActivationEpilogue CublasltActivation(
    const GemmAndBiasParams<T>* params) {
  return params->activation;
}

template <typename T, typename C_Dtype = T>
class CublasltStandardGemmProblem {
 public:
  CublasltStandardGemmProblem(
      char transa,
      char transb,
      int64_t m,
      int64_t n,
      int64_t k,
      at::opmath_type<T> alpha,
      const T* a,
      int64_t lda,
      int64_t stride_a,
      const T* b,
      int64_t ldb,
      int64_t stride_b,
      at::opmath_type<T> beta,
      C_Dtype* c,
      int64_t ldc,
      int64_t stride_c,
      int64_t batch_count,
      const void* bias,
      at::cuda::blas::GEMMAndBiasActivationEpilogue activation,
      bool set_epilogue_attribute)
      : type_info_(at::cuda::blas::detail::getCublasLtTypeInfo<T, C_Dtype>()),
        m_(m),
        n_(n),
        k_(k),
        lda_(lda),
        ldb_(ldb),
        ldc_(ldc),
        stride_a_(stride_a),
        stride_b_(stride_b),
        stride_c_(stride_c),
        batch_count_(batch_count),
        a_(a),
        b_(b),
        c_(c),
        alpha_(alpha),
        beta_(beta),
        opa_(at::cuda::blas::detail::cublasOpFromChar(transa)),
        opb_(at::cuda::blas::detail::cublasOpFromChar(transb)),
        compute_desc_(type_info_.compute_type, type_info_.scale_type),
        bias_(bias),
        activation_(activation),
        set_epilogue_attribute_(set_epilogue_attribute) {
    at::cuda::blas::detail::cublasAdjustLdLevel3(
        transa, transb, m_, n_, k_, &lda_, &ldb_, &ldc_);
    initialize();
  }

  cublasComputeType_t compute_type() const {
    return type_info_.compute_type;
  }

  cudaDataType_t scale_type() const {
    return type_info_.scale_type;
  }

  cudaDataType_t a_type() const {
    return type_info_.ab_type;
  }

  cudaDataType_t b_type() const {
    return type_info_.ab_type;
  }

  cudaDataType_t c_type() const {
    return type_info_.c_type;
  }

  cudaDataType_t d_type() const {
    return type_info_.c_type;
  }

  cublasLtMatmulDesc_t compute_desc() const {
    return compute_desc_.descriptor();
  }

  cublasLtMatrixLayout_t adesc() const {
    return adesc_->descriptor();
  }

  cublasLtMatrixLayout_t bdesc() const {
    return bdesc_->descriptor();
  }

  cublasLtMatrixLayout_t cdesc() const {
    return cdesc_->descriptor();
  }

  cublasLtMatrixLayout_t ddesc() const {
    return cdesc();
  }

  cublasLtMatrixLayout_t heuristic_bdesc() const {
    return lie_to_cublaslt_ ? fake_bdesc_->descriptor() : bdesc();
  }

  cublasLtMatrixLayout_t heuristic_cdesc() const {
    return lie_to_cublaslt_ ? fake_cdesc_->descriptor() : cdesc();
  }

  cublasLtMatrixLayout_t heuristic_ddesc() const {
    return heuristic_cdesc();
  }

  cublasLtMatmulPreference_t preference() const {
    return preference_.descriptor();
  }

  void* alpha_ptr() const {
    return alpha_ptr_;
  }

  void* beta_ptr() const {
    return beta_ptr_;
  }

  const T* a() const {
    return a_;
  }

  const T* b() const {
    return b_;
  }

  C_Dtype* c() const {
    return c_;
  }

  C_Dtype* d() const {
    return c_;
  }

  void* workspace() const {
    return workspace_.ptr;
  }

  size_t workspace_size() const {
    return workspace_.size;
  }

 private:
  void initialize() {
    alpha_ptr_ = &alpha_;
    beta_ptr_ = &beta_;
    if constexpr (std::is_same_v<T, at::Half>) {
      if (type_info_.compute_type == CUBLAS_COMPUTE_16F) {
        halpha_ = alpha_;
        hbeta_ = beta_;
        alpha_ptr_ = &halpha_;
        beta_ptr_ = &hbeta_;
      }
    }

    compute_desc_.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa_);
    compute_desc_.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb_);
    if (set_epilogue_attribute_ || bias_ != nullptr ||
        activation_ != at::cuda::blas::GEMMAndBiasActivationEpilogue::None) {
      epilogue_ =
          at::cuda::blas::detail::cublasLtEpilogue(activation_, bias_);
      compute_desc_.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue_);
    }
    if (bias_ != nullptr) {
      compute_desc_.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_);
    }

    if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
      compute_desc_.setAttribute<int32_t>(
          CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
              at::globalContext()._SMCarveout_EXPERIMENTAL().value());
    }

    if constexpr (std::is_same_v<T, at::Half>) {
      auto fp16_reduction = at::globalContext().allowFP16ReductionCuBLAS();
      if (fp16_reduction !=
          at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
        reduction_mask_ =
            fp16_reduction ==
                    at::CuBLASReductionOption::
                        DisallowReducedPrecisionAllowSplitK
                ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                   CUBLASLT_REDUCTION_SCHEME_NONE)
                : CUBLASLT_REDUCTION_SCHEME_NONE;
        preference_.setAttribute(
            CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, reduction_mask_);
      }
    } else if constexpr (std::is_same_v<T, at::BFloat16>) {
      auto bf16_reduction = at::globalContext().allowBF16ReductionCuBLAS();
      if (bf16_reduction !=
          at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK) {
        reduction_mask_ =
            bf16_reduction ==
                    at::CuBLASReductionOption::
                        DisallowReducedPrecisionAllowSplitK
                ? (CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE |
                   CUBLASLT_REDUCTION_SCHEME_NONE)
                : CUBLASLT_REDUCTION_SCHEME_NONE;
        preference_.setAttribute(
            CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, reduction_mask_);
      }
    }

    adesc_ = std::make_unique<at::cuda::blas::detail::CuBlasLtMatrixLayout>(
        type_info_.ab_type, m_, k_, lda_, opa_ != CUBLAS_OP_N);
    bdesc_ = std::make_unique<at::cuda::blas::detail::CuBlasLtMatrixLayout>(
        type_info_.ab_type, k_, n_, ldb_, opb_ != CUBLAS_OP_N);
    cdesc_ = std::make_unique<at::cuda::blas::detail::CuBlasLtMatrixLayout>(
        type_info_.c_type, m_, n_, ldc_);

    if (batch_count_ > 1) {
      int batch_as_int = static_cast<int>(batch_count_);
      adesc_->setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_as_int);
      bdesc_->setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_as_int);
      cdesc_->setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_as_int);
      adesc_->setAttribute(
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_a_);
      bdesc_->setAttribute(
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b_);
      cdesc_->setAttribute(
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_c_);
    }

    preference_.setAttribute(
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        at::cuda::blas::detail::getAlignment(reinterpret_cast<uintptr_t>(a_)));
    preference_.setAttribute(
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        at::cuda::blas::detail::getAlignment(reinterpret_cast<uintptr_t>(b_)));
    preference_.setAttribute(
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        at::cuda::blas::detail::getAlignment(reinterpret_cast<uintptr_t>(c_)));
    if (bias_ != nullptr) {
      preference_.setAttribute(
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
          at::cuda::blas::detail::getAlignment(
              reinterpret_cast<uintptr_t>(bias_)));
    }

    preference_.setAttribute(
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspace_.size);

    lie_to_cublaslt_ = reduction_mask_ == CUBLASLT_REDUCTION_SCHEME_NONE &&
        n_ == 1 && at::cuda::getCurrentDeviceProperties()->major >= 10;
    if (lie_to_cublaslt_) {
      const auto fake_ldb = ldb_ == 1 ? 2 : ldb_;
      const auto fake_ldc = ldc_ == 1 ? 2 : ldc_;
      fake_bdesc_ =
          std::make_unique<at::cuda::blas::detail::CuBlasLtMatrixLayout>(
              type_info_.ab_type, k_, 2, fake_ldb, opb_ == CUBLAS_OP_T);
      fake_cdesc_ =
          std::make_unique<at::cuda::blas::detail::CuBlasLtMatrixLayout>(
              type_info_.c_type, m_, 2, fake_ldc);
      if (batch_count_ > 1) {
        int batch_as_int = static_cast<int>(batch_count_);
        fake_bdesc_->setAttribute(
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_as_int);
        fake_cdesc_->setAttribute(
            CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_as_int);
        fake_bdesc_->setAttribute(
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_b_);
        fake_cdesc_->setAttribute(
            CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride_c_);
      }
    }
  }

  at::cuda::blas::detail::CublasLtTypeInfo<T, C_Dtype> type_info_;
  int64_t m_;
  int64_t n_;
  int64_t k_;
  int64_t lda_;
  int64_t ldb_;
  int64_t ldc_;
  int64_t stride_a_;
  int64_t stride_b_;
  int64_t stride_c_;
  int64_t batch_count_;
  const T* a_;
  const T* b_;
  C_Dtype* c_;
  at::opmath_type<T> alpha_;
  at::opmath_type<T> beta_;
  at::Half halpha_;
  at::Half hbeta_;
  void* alpha_ptr_ = nullptr;
  void* beta_ptr_ = nullptr;
  uint32_t reduction_mask_ = std::numeric_limits<uint32_t>::max();
  cublasOperation_t opa_;
  cublasOperation_t opb_;
  at::cuda::blas::detail::CuBlasLtMatmulDescriptor compute_desc_;
  at::cuda::blas::detail::CuBlasLtMatmulPreference preference_;
  const void* bias_ = nullptr;
  at::cuda::blas::GEMMAndBiasActivationEpilogue activation_;
  bool set_epilogue_attribute_;
  cublasLtEpilogue_t epilogue_ = CUBLASLT_EPILOGUE_DEFAULT;
  std::unique_ptr<at::cuda::blas::detail::CuBlasLtMatrixLayout> adesc_;
  std::unique_ptr<at::cuda::blas::detail::CuBlasLtMatrixLayout> bdesc_;
  std::unique_ptr<at::cuda::blas::detail::CuBlasLtMatrixLayout> cdesc_;
  std::unique_ptr<at::cuda::blas::detail::CuBlasLtMatrixLayout> fake_bdesc_;
  std::unique_ptr<at::cuda::blas::detail::CuBlasLtMatrixLayout> fake_cdesc_;
  at::cuda::blas::detail::CublasLtWorkspace workspace_;
  bool lie_to_cublaslt_ = false;
};

template <typename T, typename ParamsT>
auto MakeCublasltStandardGemmProblem(const ParamsT* params) {
  return CublasltStandardGemmProblem<T>(
      params->transa,
      params->transb,
      params->m,
      params->n,
      params->k,
      params->alpha,
      params->a,
      params->lda,
      CublasltStrideA(params),
      params->b,
      params->ldb,
      CublasltStrideB(params),
      CublasltBeta(params),
      params->c,
      params->ldc,
      CublasltStrideC(params),
      CublasltBatchCount(params),
      CublasltBias(params),
      CublasltActivation(params),
      true /* set_epilogue_attribute */);
}

template <typename T, typename C_Dtype>
auto MakeCublasltStandardGemmProblem(
    const GemmStridedBatchedParams<T, C_Dtype>* params) {
  return CublasltStandardGemmProblem<T, C_Dtype>(
      params->transa,
      params->transb,
      params->m,
      params->n,
      params->k,
      params->alpha,
      params->a,
      params->lda,
      params->stride_a,
      params->b,
      params->ldb,
      params->stride_b,
      params->beta,
      params->c,
      params->ldc,
      params->stride_c,
      params->batch,
      nullptr,
      at::cuda::blas::GEMMAndBiasActivationEpilogue::None,
      // Keep batched GEMM descriptor behavior aligned with the legacy path;
      // setting EPILOGUE_DEFAULT changes some cuBLASLt heuristic results.
      false /* set_epilogue_attribute */);
}


struct CublasltAlgoConfig {
  int32_t id = 0;
  uint32_t tile = 0;
  uint32_t stages = 0;
  int32_t splitk = 1;
  uint32_t reduction = 0;
  uint32_t swizzle = 0;
  uint32_t custom = 0;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12000
  uint16_t inner_shape = 0;
  uint16_t cluster_shape = 0;
#endif
};

template <typename T>
bool CublasltGetAlgoConfigAttribute(
    const cublasLtMatmulAlgo_t* algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    T* value) {
  size_t written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(
             algo, attr, value, sizeof(T), &written) == CUBLAS_STATUS_SUCCESS;
}

inline std::optional<CublasltAlgoConfig> CublasltAlgoConfigFromAlgo(
    const cublasLtMatmulAlgo_t& algo) {
  CublasltAlgoConfig config;
  bool ok =
      CublasltGetAlgoConfigAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &config.id) &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &config.tile) &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &config.stages) &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &config.splitk) &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &config.reduction) &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &config.swizzle) &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &config.custom);
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12000
  ok = ok &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &config.inner_shape) &&
      CublasltGetAlgoConfigAttribute(
          &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &config.cluster_shape);
#endif
  if (!ok) {
    return std::nullopt;
  }
  return config;
}

inline std::string CublasltAlgoConfigName(const CublasltAlgoConfig& config) {
  std::string name = fmt::sprintf(
      "Gemm_Cublaslt_id_%d_tile_%u_stages_%u_splitk_%d_red_%u_swizzle_%u_custom_%u",
      config.id,
      config.tile,
      config.stages,
      config.splitk,
      config.reduction,
      config.swizzle,
      config.custom);
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12000
  name += fmt::sprintf(
      "_inner_%hu_cluster_%hu", config.inner_shape, config.cluster_shape);
#endif
  return name;
}

inline std::optional<CublasltAlgoConfig> CublasltAlgoConfigFromName(
    const std::string& name) {
  CublasltAlgoConfig config;
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12000
  int matched = std::sscanf(
      name.c_str(),
      "Gemm_Cublaslt_id_%d_tile_%u_stages_%u_splitk_%d_red_%u_swizzle_%u_custom_%u"
      "_inner_%hu_cluster_%hu",
      &config.id,
      &config.tile,
      &config.stages,
      &config.splitk,
      &config.reduction,
      &config.swizzle,
      &config.custom,
      &config.inner_shape,
      &config.cluster_shape);
  if (matched != 9) {
    return std::nullopt;
  }
#else
  int matched = std::sscanf(
      name.c_str(),
      "Gemm_Cublaslt_id_%d_tile_%u_stages_%u_splitk_%d_red_%u_swizzle_%u_custom_%u",
      &config.id,
      &config.tile,
      &config.stages,
      &config.splitk,
      &config.reduction,
      &config.swizzle,
      &config.custom);
  if (matched != 7) {
    return std::nullopt;
  }
#endif
  return config;
}

template <typename T>
bool CublasltSetAlgoConfigAttribute(
    cublasLtMatmulAlgo_t* algo,
    cublasLtMatmulAlgoConfigAttributes_t attr,
    const T& value) {
  return cublasLtMatmulAlgoConfigSetAttribute(
             algo, attr, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS;
}

template <typename ProblemT>
bool CublasltInitializeAlgo(
    const ProblemT& problem,
    const CublasltAlgoConfig& config,
    cublasLtMatmulAlgo_t* algo) {
  auto status = cublasLtMatmulAlgoInit(
      at::cuda::getCurrentCUDABlasLtHandle(),
      problem.compute_type(),
      problem.scale_type(),
      problem.a_type(),
      problem.b_type(),
      problem.c_type(),
      problem.d_type(),
      config.id,
      algo);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return false;
  }

  bool ok =
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_TILE_ID, config.tile) &&
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, config.stages) &&
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, config.splitk) &&
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, config.reduction) &&
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, config.swizzle) &&
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, config.custom);
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 12000
  ok = ok &&
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, config.inner_shape) &&
      CublasltSetAlgoConfigAttribute(
          algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, config.cluster_shape);
#endif
  return ok;
}

template <typename ProblemT>
bool CublasltValidateAlgo(
    const ProblemT& problem,
    cublasLtMatmulAlgo_t* algo) {
  cublasLtMatmulHeuristicResult_t check_result = {};
  auto status = cublasLtMatmulAlgoCheck(
      at::cuda::getCurrentCUDABlasLtHandle(),
      problem.compute_desc(),
      problem.adesc(),
      problem.bdesc(),
      problem.cdesc(),
      problem.ddesc(),
      algo,
      &check_result);
  return status == CUBLAS_STATUS_SUCCESS &&
      check_result.workspaceSize <= problem.workspace_size();
}

template <typename T>
struct CublasltStandardGemmProblemFactory {
  template <typename ParamsT>
  auto operator()(const ParamsT* params) const {
    return MakeCublasltStandardGemmProblem<T>(params);
  }
};


template <typename ParamsT, typename ProblemFactory>
class CublasltMatmulOp : public Callable<ParamsT> {
 public:
  explicit CublasltMatmulOp(std::string name)
      : name_(std::move(name)), config_(CublasltAlgoConfigFromName(name_)) {}

  CublasltMatmulOp(std::string name, CublasltAlgoConfig config)
      : name_(std::move(name)), config_(config) {}

  TuningStatus Call(const ParamsT* params) override {
    auto problem = problem_factory_(params);
    if (problem.workspace() == nullptr) {
      return FAIL;
    }
    cublasLtMatmulAlgo_t algo = {};
    if (!InitializeAlgo(problem, &algo)) {
      return FAIL;
    }
    auto status = cublasLtMatmul(
        at::cuda::getCurrentCUDABlasLtHandle(),
        problem.compute_desc(),
        problem.alpha_ptr(),
        problem.a(),
        problem.adesc(),
        problem.b(),
        problem.bdesc(),
        problem.beta_ptr(),
        problem.c(),
        problem.cdesc(),
        problem.d(),
        problem.ddesc(),
        &algo,
        problem.workspace(),
        problem.workspace_size(),
        at::cuda::getCurrentCUDAStream());
    return status == CUBLAS_STATUS_SUCCESS ? OK : FAIL;
  }

 private:
  template <typename ProblemT>
  bool InitializeAlgo(const ProblemT& problem, cublasLtMatmulAlgo_t* algo) {
    if (!config_.has_value() ||
        !CublasltInitializeAlgo(problem, config_.value(), algo) ||
        !CublasltValidateAlgo(problem, algo)) {
      return false;
    }
    return true;
  }

  ProblemFactory problem_factory_;
  std::string name_;
  std::optional<CublasltAlgoConfig> config_;
};

template <typename ProblemT>
std::vector<std::pair<std::string, CublasltAlgoConfig>>
GetCublasltHeuristicCandidates(const ProblemT& problem) {
  std::vector<std::pair<std::string, CublasltAlgoConfig>> ret;
  if (problem.workspace() == nullptr) {
    return ret;
  }

  std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
      getTuningContext()->GetCublasLtRequestedAlgoCount());
  int returned_result = 0;
  auto status = cublasLtMatmulAlgoGetHeuristic(
      at::cuda::getCurrentCUDABlasLtHandle(),
      problem.compute_desc(),
      problem.adesc(),
      problem.heuristic_bdesc(),
      problem.heuristic_cdesc(),
      problem.heuristic_ddesc(),
      problem.preference(),
      static_cast<int>(heuristic_results.size()),
      heuristic_results.data(),
      &returned_result);
  if (status != CUBLAS_STATUS_SUCCESS || returned_result == 0) {
    return ret;
  }

  for (int i = 0; i < returned_result; ++i) {
    if (heuristic_results[i].state != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    auto config = CublasltAlgoConfigFromAlgo(heuristic_results[i].algo);
    if (!config.has_value()) {
      continue;
    }
    ret.emplace_back(CublasltAlgoConfigName(config.value()), config.value());
  }
  return ret;
}

} // anonymous namespace

template <typename ParamsT, typename ProblemFactory>
class CublasltMatmulTunableOp : public TunableOp<ParamsT> {
 public:
  CublasltMatmulTunableOp() = default;

 protected:
  void RegisterOpCandidates(const ParamsT* params) override {
    const auto params_sig = params->Signature();
    std::scoped_lock l{candidate_lock_};
    if (candidate_names_by_params_.find(params_sig) !=
        candidate_names_by_params_.end()) {
      return;
    }

    std::vector<std::string> candidate_names{"Default"};
    auto problem = problem_factory_(params);
    for (auto&& [name, config] : GetCublasltHeuristicCandidates(problem)) {
      if (!this->HasOp(name)) {
        this->RegisterOp(
            name,
            std::make_unique<CublasltMatmulOp<ParamsT, ProblemFactory>>(
                name, config));
      }
      candidate_names.emplace_back(std::move(name));
    }
    candidate_names_by_params_.emplace(params_sig, std::move(candidate_names));
  }

  std::vector<std::string> CandidateNames(const ParamsT* params) const override {
    const auto params_sig = params->Signature();
    std::scoped_lock l{candidate_lock_};
    auto it = candidate_names_by_params_.find(params_sig);
    if (it == candidate_names_by_params_.end()) {
      return {"Default"};
    }
    return it->second;
  }

  bool RegisterOpForResult(
      const ResultEntry& result,
      const ParamsT* /*params*/) override {
    const auto name = result.GetKey();
    if (this->HasOp(name)) {
      return true;
    }
    if (!CublasltAlgoConfigFromName(name).has_value()) {
      return false;
    }
    this->RegisterOp(
        name,
        std::make_unique<CublasltMatmulOp<ParamsT, ProblemFactory>>(name));
    return true;
  }

 private:
  ProblemFactory problem_factory_;
  mutable std::mutex candidate_lock_;
  std::unordered_map<std::string, std::vector<std::string>>
      candidate_names_by_params_;
};

template <typename T, typename ParamsT>
class CublasltGemmTunableOp
    : public CublasltMatmulTunableOp<
          ParamsT,
          CublasltStandardGemmProblemFactory<T>> {};


} // namespace at::cuda::tunable

#endif // USE_ROCM
