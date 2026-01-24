#include <ATen/Config.h>

#include <ATen/Context.h>

#include <c10/core/CPUAllocator.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

#include <ATen/cpu/FlushDenormal.h>

#ifdef USE_FBGEMM
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
#include <fbgemm/Fbgemm.h>
C10_DIAGNOSTIC_POP()
#endif // USE_FBGEMM
#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <cpuinfo.h>
#endif
namespace at {

/*
  These const variables defined the fp32 precisions for different backend
  We have "generic", "cuda", "mkldnn" backend now and we can choose fp32
  prevision from "ieee", "tf32", "bf16" and "none". The "ieee" precision means
  IEEE standard floating point format, "tf32" and "bf16" means we are allowed to
  use "tf32" or "bf16" as internal computation data types for fp32 computations.
  And "none" means it is override-able by parent's node

  generic->mkldnn->matmul
                ->conv
                ->rnn
         ->cuda ->matmul
                ->conv
                ->rnn
*/

Float32Backend str2backend(const std::string& name) {
  if (name == "generic")
    return Float32Backend::GENERIC;
  else if (name == "cuda")
    return Float32Backend::CUDA;
  else if (name == "mkldnn")
    return Float32Backend::MKLDNN;
  TORCH_CHECK(false, "Unknown backend: ", name);
}

Float32Op str2op(const std::string& name) {
  if (name == "all")
    return Float32Op::ALL;
  else if (name == "conv")
    return Float32Op::CONV;
  else if (name == "rnn")
    return Float32Op::RNN;
  else if (name == "matmul")
    return Float32Op::MATMUL;
  TORCH_CHECK(false, "Unknown op: ", name);
}

Float32Precision str2precision(const std::string& name) {
  if (name == "none")
    return Float32Precision::NONE;
  else if (name == "ieee")
    return Float32Precision::IEEE;
  else if (name == "tf32")
    return Float32Precision::TF32;
  else if (name == "bf16")
    return Float32Precision::BF16;
  TORCH_CHECK(false, "Unknown precision: ", name);
}

std::string precision2str(Float32Precision prec) {
  switch (prec) {
    case Float32Precision::NONE:
      return "none";
    case Float32Precision::IEEE:
      return "ieee";
    case Float32Precision::TF32:
      return "tf32";
    case Float32Precision::BF16:
      return "bf16";
  }
  TORCH_CHECK(false, "Invalid enum Float32Precision(", static_cast<int>(prec), ")");
}

#ifdef USE_ROCM
static constexpr const auto rocm_allow_group_gemm_ck = "ROCM_ALLOW_GROUP_GEMM_CK";
#endif

Context::Context() = default;

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
Context& globalContext() {
  static Context globalContext_;
  return globalContext_;
}

// NB: This method is *purely* whether or not a user requested
// that CuDNN was enabled, it doesn't actually say anything about
// whether or not CuDNN is actually usable.
bool Context::userEnabledCuDNN() const {
  return enabled_cudnn;
}

void Context::setUserEnabledCuDNN(bool e) {
  enabled_cudnn = e;
}

bool Context::userEnabledMkldnn() const {
  return enabled_mkldnn;
}

void Context::setUserEnabledMkldnn(bool e) {
  enabled_mkldnn = e;
}

bool Context::deterministicCuDNN() const {
  return deterministic_cudnn;
}

void Context::setDeterministicCuDNN(bool b) {
  deterministic_cudnn = b;
}

bool Context::deterministicMkldnn() const {
  return deterministic_mkldnn;
}

void Context::setDeterministicMkldnn(bool b) {
  deterministic_mkldnn = b;
}

bool Context::deterministicAlgorithms() const {
  return _deterministic_algorithms;
}

bool Context::deterministicAlgorithmsWarnOnly() const {
  return _deterministic_algorithms_warn_only;
}

void Context::setDeterministicAlgorithms(bool b, bool warn_only = false) {
  _deterministic_algorithms = b;
  _deterministic_algorithms_warn_only = warn_only;
}

bool Context::deterministicFillUninitializedMemory() const {
  return _deterministic_fill_uninitialized_memory;
}

void Context::setDeterministicFillUninitializedMemory(bool b) {
  _deterministic_fill_uninitialized_memory = b;
}

void Context::alertNotDeterministic(std::string_view const& caller) {
  if (globalContext().deterministicAlgorithms()) {
    if (globalContext().deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN(
        caller, " does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'. "
        "You can file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this operation.");
    } else {
      TORCH_CHECK(false,
        caller, " does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'. You can turn off "
        "determinism just for this operation, or you can use the "
        "'warn_only=True' option, if that's acceptable for your application. "
        "You can also file an issue at https://github.com/pytorch/pytorch/issues "
        "to help us prioritize adding deterministic support for this operation.");
    }
  }
}

bool Context::userEnabledNNPACK() const {
  return enabled_nnpack;
}

void Context::setUserEnabledNNPACK(bool e) {
  enabled_nnpack = e;
}

bool Context::allowTF32CuDNN(std::optional<Float32Op> op) const {
  if (!op.has_value()) {
    bool allow_tf32_rnn = float32Precision(Float32Backend::CUDA, Float32Op::RNN) == Float32Precision::TF32;
    bool allow_tf32_conv = float32Precision(Float32Backend::CUDA, Float32Op::CONV) == Float32Precision::TF32;
    TORCH_CHECK(
        allow_tf32_rnn == allow_tf32_conv && allow_tf32_rnn == allow_tf32_cudnn,
        "PyTorch is checking whether allow_tf32 is enabled for cuDNN without a specific operator name,",
        "but the current flag(s) indicate that cuDNN conv and cuDNN RNN have different TF32 flags.",
        "This combination indicates that you have used a mix of the legacy and new APIs to set the TF32 flags. ",
        "We suggest only using the new API to set the TF32 flag(s). See also: ",
        "https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices");
  } else {
    return float32Precision(Float32Backend::CUDA, op.value()) == Float32Precision::TF32;
  }
  return allow_tf32_cudnn;
}

void Context::setAllowTF32CuDNN(bool b) {
  setFloat32Precision(Float32Backend::CUDA, Float32Op::RNN, b ? Float32Precision::TF32 : Float32Precision::NONE);
  setFloat32Precision(Float32Backend::CUDA, Float32Op::CONV, b ? Float32Precision::TF32 : Float32Precision::NONE);
  allow_tf32_cudnn = b;
}

void Context::setSDPPriorityOrder(const std::vector<int64_t>& order) {
  // TODO*eqy): should it always be the number of backends - 1 (override backend excluded?)
  TORCH_CHECK(at::num_sdp_backends == sdp_priority_order.size(),
    "setSDPPriority order expected ", sdp_priority_order.size() - 1, " but got ",
    at::num_sdp_backends, " unique backends specified in priority order.");
  for (uint32_t i = 0; i < order.size(); i++) {
    sdp_priority_order[i] = static_cast<at::SDPBackend>(order[i]);
  }
}

std::array<at::SDPBackend, at::num_sdp_backends> Context::sDPPriorityOrder() {
  return sdp_priority_order;
}

bool Context::allowTF32OneDNN() const {
  return allow_tf32_onednn;
}

  // NOLINTNEXTLINE(clang-diagnostic-unused-parameter)
  void Context::setAllowTF32OneDNN(bool b){
  #ifdef USE_XPU
  allow_tf32_onednn = b;
  #else
  TORCH_WARN("TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support.");
  #endif
}

#ifdef USE_ROCM
bool Context::rocmAllowGroupGemmCk() const {
    const auto allow_group_gemm_ck = c10::utils::check_env(rocm_allow_group_gemm_ck) == true;
    return allow_group_gemm_ck;
}
#endif

bool Context::userEnabledFlashSDP() const {
  return enabled_flashSDP;
}

void Context::setSDPUseFlash(bool e) {
  enabled_flashSDP = e;
}

bool Context::userEnabledFA3SDP() const {
  return enabled_fa3SDP;
}

void Context::setSDPUseFA3(bool e) {
  enabled_fa3SDP = e;
}

bool Context::userEnabledMemEfficientSDP() const {
  return enabled_mem_efficientSDP;
}

void Context::setSDPUseMemEfficient(bool e) {
  enabled_mem_efficientSDP = e;
}

bool Context::userEnabledMathSDP() const {
  return enabled_mathSDP;
}

void Context::setSDPUseMath(bool e) {
  enabled_mathSDP = e;
}

bool Context::allowFP16BF16ReductionMathSDP() const {
  return allow_fp16_bf16_reduction_mathSDP;
}

void Context::setAllowFP16BF16ReductionMathSDP(bool e) {
  allow_fp16_bf16_reduction_mathSDP = e;
}

bool Context::userEnabledCuDNNSDP() const {
  return enabled_cudnnSDP;
}

void Context::setSDPUseCuDNN(bool e) {
  enabled_cudnnSDP = e;
}

void Context::setSDPUseOverrideable(bool e) {
  enabled_overrideable = e;
}

bool Context::userEnabledOverrideableSDP() const {
  return enabled_overrideable;
}

bool Context::benchmarkCuDNN() const {
  return benchmark_cudnn;
}

void Context::setBenchmarkCuDNN(bool b) {
  benchmark_cudnn = b;
}

int Context::benchmarkLimitCuDNN() const {
  return benchmark_limit_cudnn;
}

void Context::setBenchmarkLimitCuDNN(int b) {
  benchmark_limit_cudnn = b;
}

bool Context::immediateMiopen() const {
  return immediate_miopen;
}

void Context::setImmediateMiopen(bool b) {
  immediate_miopen = b;
}

bool Context::allowTF32CuBLAS() const {
  bool legacy_allow_tf32 = float32_matmul_precision != at::Float32MatmulPrecision::HIGHEST;
  bool allow_tf32_new = float32Precision(Float32Backend::CUDA, Float32Op::MATMUL) == Float32Precision::TF32;
  TORCH_CHECK(
      legacy_allow_tf32 == allow_tf32_new,
      "PyTorch is checking whether allow_tf32_new is enabled for cuBlas matmul,",
      "Current status indicate that you have used mix of the legacy and new APIs to set the TF32 status for cublas matmul. ",
      "We suggest only using the new API to set the TF32 flag. See also: ",
      "https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices");
  return allow_tf32_new;
}

void Context::setAllowTF32CuBLAS(bool b) {
  float32_matmul_precision = b ? at::Float32MatmulPrecision::HIGH : at::Float32MatmulPrecision::HIGHEST;
  setFloat32Precision(Float32Backend::CUDA, Float32Op::MATMUL, b ? Float32Precision::TF32 : Float32Precision::IEEE);
}

Float32MatmulPrecision Context::float32MatmulPrecision() const {
  bool invalid = float32Precision(Float32Backend::CUDA, Float32Op::MATMUL) == Float32Precision::TF32 &&
      float32_matmul_precision == at::Float32MatmulPrecision::HIGHEST;
  invalid = invalid ||
      (float32Precision(Float32Backend::MKLDNN, Float32Op::MATMUL) == Float32Precision::BF16 &&
       float32_matmul_precision != at::Float32MatmulPrecision::MEDIUM);
  invalid = invalid ||
      (float32Precision(Float32Backend::MKLDNN, Float32Op::MATMUL) == Float32Precision::TF32 &&
       float32_matmul_precision != at::Float32MatmulPrecision::HIGH);
  TORCH_CHECK(
      !invalid,
      "PyTorch is checking the matmul precision without a specific backend name,",
      "Current status indicate that you have used mix of the legacy and new APIs to set the matmul precision. ",
      "We suggest only using the new API for matmul precision. See also: ",
      "https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices");
  return float32_matmul_precision;
}

Float32Precision Context::float32Precision(Float32Backend backend, Float32Op op) const {
  std::pair<Float32Backend, Float32Op> key{backend, op};
  auto it = fp32_precision.find(key);
  TORCH_CHECK(it != fp32_precision.end(), "Invalid (backend, op) pair: (", backend, ", ", op, ")");

  Float32Precision precision = it->second;
  if (precision == Float32Precision::NONE) {
    key.second = Float32Op::ALL;
    precision = fp32_precision.find(key)->second;
  }
  if (precision == Float32Precision::NONE) {
    key.first = Float32Backend::GENERIC;
    precision = fp32_precision.find(key)->second;
  }

  // "cuda" does not support "bf16"
  if (backend == Float32Backend::CUDA && precision == Float32Precision::BF16) {
    return Float32Precision::NONE;
  }
  return precision;
}

void Context::setFloat32MatmulPrecision(const std::string &s) {
  auto match = [this](const std::string & s_) {
    // TODO: consider if CuDNN field needs to also be set for potential future CuDNN ops like multi-headed attention
    if (s_ == "highest") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGHEST;
      setFloat32Precision(Float32Backend::CUDA, Float32Op::MATMUL, Float32Precision::IEEE);
      setFloat32Precision(Float32Backend::MKLDNN, Float32Op::MATMUL, Float32Precision::IEEE);
      return true;
    } else if (s_ == "high") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGH;
      setFloat32Precision(Float32Backend::CUDA, Float32Op::MATMUL, Float32Precision::TF32);
      setFloat32Precision(Float32Backend::MKLDNN, Float32Op::MATMUL, Float32Precision::TF32);
      return true;
    } else if (s_ == "medium") {
      float32_matmul_precision = at::Float32MatmulPrecision::MEDIUM;
      setFloat32Precision(Float32Backend::CUDA, Float32Op::MATMUL, Float32Precision::TF32);
      setFloat32Precision(Float32Backend::MKLDNN, Float32Op::MATMUL, Float32Precision::BF16);
      return true;
    }
    return false;
  };
  if (match(s)) { return; }
  std::string sl;
  std::transform(s.begin(), s.end(), sl.begin(),
                 [](unsigned char c) -> unsigned char { return std::tolower(c); });
  if (match(sl)) { return; }
  TORCH_WARN(s, " is not one of 'highest', 'high', or 'medium'; the current"
    "setFloat32MatmulPrecision call has no effect.");
}

void Context::setFloat32Precision(Float32Backend backend, Float32Op op, Float32Precision p) {
  auto it = fp32_precision.find(std::make_pair(backend, op));
  TORCH_CHECK(
      it != fp32_precision.end(),
      "Invalid (backend, op) pair: (", backend, ", ", op, ")");
  TORCH_CHECK(
      !(backend == Float32Backend::CUDA && p == Float32Precision::BF16),
      "backend 'cuda' does not support precision 'bf16'");

  it->second = p;
}

at::LinalgBackend Context::linalgPreferredBackend() const {
  return linalg_preferred_backend;
}

void Context::setLinalgPreferredBackend(at::LinalgBackend b) {
  linalg_preferred_backend = b;
  TORCH_CHECK((b != at::LinalgBackend::Cusolver) || hasCuSOLVER(),
      "Cannot set preferred backend to cuSOLVER if PyTorch has not been compiled with cuSOLVER.");
  TORCH_CHECK((b != at::LinalgBackend::Magma) || hasMAGMA(),
      "Cannot set preferred backend to MAGMA if PyTorch has not been compiled with MAGMA.");
  if (b != at::LinalgBackend::Default) {
    TORCH_WARN_ONCE(
      "torch.backends.cuda.preferred_linalg_library is an experimental feature. "
      "If you see any error or unexpected behavior when this flag is set "
      "please file an issue on GitHub."
    );
  }
}

at::BlasBackend Context::blasPreferredBackend() {
  // Rather than put logic for interpreting what Default means at every
  // call site for blasPreferredBackend(), we set it to an actual value.
  if (blas_preferred_backend == at::BlasBackend::Default) {
    blas_preferred_backend = at::BlasBackend::Cublas;
    // This logic sits in the getter because it needs to validate
    // values set via env vars such as TORCH_BLAS_PREFER_CUBLASLT
    // which initialize the backend without calling the setter
#ifdef USE_ROCM
    // AMD Instinct targets prefer hipblaslt
    static const bool hipblaslt_preferred = []() {
      const auto& archs = detail::getCUDAHooks().getHipblasltPreferredArchs();
      for (auto index: c10::irange(detail::getCUDAHooks().deviceCount())) {
        if (!detail::getCUDAHooks().isGPUArch(archs, index)) {
          return false;
        }
      }
      return true;
    }();
    if (hipblaslt_preferred) {
      blas_preferred_backend = at::BlasBackend::Cublaslt;
    }
#endif
  }

#ifdef USE_ROCM
  // hipblaslt support for all archs is not as complete as hipblas
  if (blas_preferred_backend == at::BlasBackend::Cublaslt) {
    static const bool hipblaslt_unsupported = []() {
      if(!hasCuBLASLt())
      {
          return true;
      }
      const auto& archs = detail::getCUDAHooks().getHipblasltSupportedArchs();
      for (auto index: c10::irange(detail::getCUDAHooks().deviceCount())) {
        if (!detail::getCUDAHooks().isGPUArch(archs, index)) {
          TORCH_WARN_ONCE(
            "Attempting to use hipBLASLt on an unsupported architecture! "
            "Overriding blas backend to hipblas");
          return true;
        }
      }
      return false;
    }();
    if (hipblaslt_unsupported) blas_preferred_backend = at::BlasBackend::Cublas;
  }
#endif
  return blas_preferred_backend;
}

bool Context::ckSupported() {
#ifdef USE_ROCM
  static const std::vector<std::string> supported_archs = {
    "gfx90a", "gfx942", "gfx950"
  };
  for (auto index : c10::irange(detail::getCUDAHooks().deviceCount())) {
    if(!detail::getCUDAHooks().isGPUArch(supported_archs, index)) {
      TORCH_WARN_ONCE(
        "Attempting to use CK on an unsupported architecture! Cannot set backend to CK");
      return false;
    }
  }
  return true;
#else
  return false;
#endif
}

void Context::setBlasPreferredBackend(at::BlasBackend b) {
#ifdef _MSC_VER
  TORCH_WARN_ONCE(
    "torch.backends.cuda.preferred_blas_library is an experimental feature. "
    "It is not supported on Windows."
  );
#else
  TORCH_CHECK((b != at::BlasBackend::Cublaslt) || hasCuBLASLt(),
      "Cannot set preferred backend to cuBLASLt if PyTorch has not been compiled with cuBLASLt.");
#ifdef USE_ROCM
  static const bool ckSupportedFlag = ckSupported();
  static const bool hasCKGEMMFlag = hasCKGEMM();
  TORCH_CHECK((b != at::BlasBackend::Ck) || (ckSupportedFlag && hasCKGEMMFlag),
      "Cannot set preferred blas backend to CK since following conditions are not true: ",
      "architecture supported for CK: ", ckSupportedFlag,
      ", PyTorch built with CK GEMM support: ", hasCKGEMMFlag);
#endif
  if (b != at::BlasBackend::Default && b != at::BlasBackend::Cublas) {
    TORCH_WARN_ONCE(
      "torch.backends.cuda.preferred_blas_library is an experimental feature. "
      "If you see any error or unexpected behavior when this flag is set "
      "please file an issue on GitHub."
    );
  }
  blas_preferred_backend = b;
#endif
}

at::ROCmFABackend Context::getROCmFAPreferredBackend() {
#ifdef USE_ROCM
  // Set potential "Default" value so we don't have to interpret at call sites.
  // We use aotriton backend as the default, for now.
  if(rocm_fa_preferred_backend == at::ROCmFABackend::Default) {
    rocm_fa_preferred_backend = at::ROCmFABackend::AOTriton;
  } else if (rocm_fa_preferred_backend == at::ROCmFABackend::Ck) {
    // This logic sits in the getter because it needs to validate
    // values set via env vars such as TORCH_ROCM_FA_PREFER_CK
    // which initialize the backend without calling the setter
    // Perform validity checking
    static const bool hasCKSDPAFlag = hasCKSDPA();
    static const bool ckSupportedFlag = ckSupported();
    if(!(hasCKSDPAFlag && ckSupportedFlag)){
      TORCH_WARN_ONCE(
        "Cannot set preferred SDPA backend to CK since following conditions are not true: ",
        "architecture supported for CK: ", ckSupportedFlag,
        ", PyTorch built with CK SDPA support: ", hasCKSDPAFlag);
      rocm_fa_preferred_backend = at::ROCmFABackend::AOTriton;
    }
  }
#endif

  return rocm_fa_preferred_backend;
}

void Context::setROCmFAPreferredBackend(at::ROCmFABackend b) {
#ifdef USE_ROCM
  static const bool hasCKSDPAFlag = hasCKSDPA();
  static const bool ckSupportedFlag = ckSupported();
  TORCH_CHECK((b != at::ROCmFABackend::Ck) || (hasCKSDPAFlag && ckSupportedFlag),
      "Cannot set preferred SDPA backend to CK since following conditions are not true: ",
      "architecture supported for CK: ", ckSupportedFlag,
      ", PyTorch built with CK SDPA support: ", hasCKSDPAFlag);
#endif
  rocm_fa_preferred_backend = b;
}

CuBLASReductionOption Context::allowFP16ReductionCuBLAS() const {
  return allow_fp16_reduction_cublas;
}

CuBLASReductionOption inline get_reduction_option(bool allow_reduced_precision, bool allow_splitk) {
  TORCH_CHECK(
      !(allow_reduced_precision && !allow_splitk),
      "allow_splitk=False is not supported when reduced precision reductions are enabled");
  if (allow_reduced_precision) {
    return CuBLASReductionOption::AllowReducedPrecisionWithSplitK;
  } else if (allow_splitk) {
    return CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK;
  } else {
    return CuBLASReductionOption::DisallowReducedPrecisionDisallowSplitK;
  }
}

void Context::setAllowFP16ReductionCuBLAS(bool allow_reduced_precision, bool allow_splitk) {
  allow_fp16_reduction_cublas = get_reduction_option(allow_reduced_precision, allow_splitk);
}

CuBLASReductionOption Context::allowBF16ReductionCuBLAS() const {
  return allow_bf16_reduction_cublas;
}

void Context::setAllowBF16ReductionCuBLAS(bool allow_reduced_precision, bool allow_splitk) {
  allow_bf16_reduction_cublas = get_reduction_option(allow_reduced_precision, allow_splitk);
}

bool Context::allowFP16AccumulationCuBLAS() const {
  return allow_fp16_accumulation_cublas;
}

void Context::setAllowFP16AccumulationCuBLAS(bool b) {
  allow_fp16_accumulation_cublas = b;
}

std::optional<int32_t> Context::_SMCarveout_EXPERIMENTAL() const {
  return sm_carveout;
}

void Context::_setSMCarveout_EXPERIMENTAL(std::optional<int32_t> c) {
  if (c.has_value()) {
    TORCH_WARN_ONCE(
      "Setting the SM carveout for matmuls is a temporary experimental mitigation for performance issues, "
      "while more robust solutions are developed. It may be removed at any moment without notice.");
  }
  sm_carveout = c;
}

bool Context::hasMKL() {
#if AT_MKL_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasMKLDNN() {
#if AT_MKLDNN_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasKleidiAI() {
  return AT_KLEIDIAI_ENABLED();
}

bool Context::hasOpenMP() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

bool Context::hasLAPACK() {
#if AT_BUILD_WITH_LAPACK()
  return true;
#else
  return false;
#endif
}

bool Context::hasEigenSparse() {
#if AT_USE_EIGEN_SPARSE()
  return true;
#else
  return false;
#endif
}

at::QEngine Context::qEngine() const {
  static auto _quantized_engine = []() {
    at::QEngine qengine = at::kNoQEngine;
#if defined(C10_MOBILE) && defined(USE_PYTORCH_QNNPACK)
    qengine = at::kQNNPACK;
#endif

#if AT_MKLDNN_ENABLED()
    qengine = at::kONEDNN;
#endif

#ifdef USE_FBGEMM
    if (fbgemm::fbgemmSupportedCPU()) {
      /* X86 is enabled if and only if fbgemm is available.
       * It combines goodness of fbgemm and onednn by dispatching.
       * If onednn not available, always dispatch to fbgemm.
       * Make it default qengine for X86 CPU platforms.
      */
      qengine = at::kX86;
    }
#endif
    return qengine;
  }();
  auto qt_engine = quantized_engine.load();
  return qt_engine == at::QEngine::NoQEngine ? _quantized_engine : qt_engine;
}

void Context::setQEngine(at::QEngine e) {
  const auto& qengines = supportedQEngines();
  if (std::find(qengines.begin(), qengines.end(), e) != qengines.end()) {
    quantized_engine.store(e);
    return;
  }
  TORCH_CHECK(false, "quantized engine ", toString(e), " is not supported");
}

const std::vector<at::QEngine>& Context::supportedQEngines() {
  static auto supported_qengines = []() {
    std::vector<at::QEngine> engines = {};
    // Engines are listed in priority order: later one wins
    // By default we prefer FBGEMM if we're running on server side
    // QNNPACK on server side has some issue, so we disable it by default.
#ifdef USE_PYTORCH_QNNPACK
    engines.push_back(at::kQNNPACK);
#endif

#if AT_MKLDNN_ENABLED()
    engines.push_back(at::kONEDNN);
#endif

#ifdef USE_FBGEMM
    if (fbgemm::fbgemmSupportedCPU()) {
      engines.push_back(at::kX86);
      // The X86 qengine is available if and only if FBGEMM is available
      engines.push_back(at::kFBGEMM);
    }
#endif

    return engines;
  }();
  return supported_qengines;
}

bool Context::isXNNPACKAvailable() {
#ifdef USE_XNNPACK
  return true;
#else
  return false;
#endif
}

void Context::setCheckSparseTensorInvariants(bool e) {
  enable_sparse_tensor_invariant_checks = e;
}

bool Context::checkSparseTensorInvariants() const {
  return enable_sparse_tensor_invariant_checks;
}

bool Context::releaseWeightsWhenPrepacking() const {
  return release_original_weights;
}

void Context::setReleaseWeightsWhenPrepacking(bool e) {
  release_original_weights = e;
}

bool Context::setFlushDenormal(bool on) {
  return at::cpu::set_flush_denormal(on);
}

Allocator* getCPUAllocator() {
  return c10::GetCPUAllocator();
}

// override_allow_tf32_flag = true
//    means the allow_tf32 flags are overridden and tf32 is force disabled
// override_allow_tf32_flag = false
//    means the original allow_tf32 flags are followed
thread_local static bool override_allow_tf32_flag = false;

NoTF32Guard::NoTF32Guard() {
  if (!override_allow_tf32_flag) {
    changed = true;
    override_allow_tf32_flag = true;
  }
}

NoTF32Guard::~NoTF32Guard() {
  if (changed) {
    override_allow_tf32_flag = false;
  }
}

bool NoTF32Guard::should_disable_tf32() {
  return override_allow_tf32_flag;
}

// Ops can query this flag to know they are in the backward pass.
// This information can be used, for example, to select implementations
// with different numerical or performance characteristics.
// See https://pytorch.org/docs/stable/notes/numerical_accuracy.html for details.
thread_local static bool rocm_is_backward_pass;

ROCmBackwardPassGuard::ROCmBackwardPassGuard() {
  rocm_is_backward_pass = true;
}

ROCmBackwardPassGuard::~ROCmBackwardPassGuard() {
  rocm_is_backward_pass = false;
}

bool ROCmBackwardPassGuard::is_backward_pass() {
  return rocm_is_backward_pass;
}

bool Context::areVmapFallbackWarningsEnabled() const {
  return display_vmap_fallback_warnings_;
}

void Context::setDisplayVmapFallbackWarnings(bool enabled) {
  display_vmap_fallback_warnings_ = enabled;
}

bool Context::warnOnAccumulateGradStreamMismatch() const {
  return warn_on_accumulate_grad_stream_mismatch_;
}

void Context::setWarnOnAccumulateGradStreamMismatch(bool enabled) {
  warn_on_accumulate_grad_stream_mismatch_ = enabled;
}

bool Context::isDefaultMobileCPUAllocatorSet() {
  return prev_allocator_ptr_ != nullptr;
}

void Context::setDefaultMobileCPUAllocator() {
  TORCH_CHECK(prev_allocator_ptr_ == nullptr,
      "Already within the scope of another non-default cpu allocator."
      "Cannot set another allocator.");
  // Setting the priority high to make sure no other allocator gets used instead of this.
  prev_allocator_ptr_ = c10::GetCPUAllocator();
  c10::SetCPUAllocator(c10::GetDefaultMobileCPUAllocator(), /*priority*/ 100);
}

void Context::unsetDefaultMobileCPUAllocator() {
  TORCH_CHECK(prev_allocator_ptr_ != nullptr,
      "setDefaultMobileCPUAllocator must have been called "
      "before unsetDefaultMobileCPUAllocator.");
  // Setting the priority high to make sure no other allocator gets used instead of this.
  c10::SetCPUAllocator(prev_allocator_ptr_ , /*priority*/ 100);
  prev_allocator_ptr_ = nullptr;
}

bool Context::allowFP16ReductionCPU() const {
  return allow_fp16_reduction_cpu;
}

void Context::setAllowFP16ReductionCPU(bool b) {
  if ( b && !allow_fp16_reduction_cpu) {
    // Check that CPU supports fp16 reductions
#if defined(__aarch64__) && !defined(C10_MOBILE)
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_fp16_arith())
#else
    // NOLINTNEXTLINE(facebook-hte-MissingBraces)
    if (true)
#endif
      TORCH_CHECK(false, "Float16 arithmetic is not supported by the CPU!");
  }
  allow_fp16_reduction_cpu = b;
}
} // namespace at
