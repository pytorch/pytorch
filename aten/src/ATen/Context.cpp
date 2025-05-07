#include <ATen/Config.h>

#include <ATen/Context.h>

#include <c10/core/CPUAllocator.h>
#include <c10/util/Logging.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

#include <ATen/cpu/FlushDenormal.h>

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#endif // USE_FBGEMM
#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <cpuinfo.h>
#endif

namespace at {

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

bool Context::allowTF32CuDNN() const {
  return allow_tf32_cudnn;
}

void Context::setAllowTF32CuDNN(bool b) {
  allow_tf32_cudnn = b;
}

void Context::setSDPPriorityOrder(const std::vector<int64_t>& order) {
  // TODO*eqy): should it always be the number of backends - 1 (override backend excluded?)
  TORCH_CHECK(at::num_sdp_backends == sdp_priority_order.size(),
    "setSDPPriority order expected ", sdp_priority_order.size() - 1, " but got ",
    at::num_sdp_backends, " unique backends specified in priority order.");
  for (uint32_t i = 0; i < order.size(); i++) {
    sdp_priority_order[i] = (at::SDPBackend) order[i];
  }
}

std::array<at::SDPBackend, at::num_sdp_backends> Context::sDPPriorityOrder() {
  return sdp_priority_order;
}

bool Context::allowTF32OneDNN() const {
  return allow_tf32_onednn;
}

void Context::setAllowTF32OneDNN(bool b){
#ifdef USE_XPU
  allow_tf32_onednn = b;
#else
  TORCH_WARN("TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support.");
#endif
}

bool Context::userEnabledFlashSDP() const {
  return enabled_flashSDP;
}

void Context::setSDPUseFlash(bool e) {
  enabled_flashSDP = e;
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

static constexpr const auto cublas_config_var_name = "CUBLAS_WORKSPACE_CONFIG";
static constexpr const std::array<const char*, 2> cublas_deterministic_configs = {":4096:8", ":16:8"};
#ifdef USE_ROCM
static constexpr const auto hipblaslt_allow_tf32 = "HIPBLASLT_ALLOW_TF32";
#endif

bool Context::checkCuBLASConfigDeterministic() {
  // If using CUDA 10.2 or greater, need to make sure CuBLAS workspace config
  // is set to deterministic setting
  if (hasCUDART()) {
    const auto workspace_config = c10::utils::get_env(cublas_config_var_name);
    return (workspace_config == cublas_deterministic_configs[0] || workspace_config == cublas_deterministic_configs[1]);
  }
  return true;
}

void Context::alertCuBLASConfigNotDeterministic() const {
  static const bool cublas_config_deterministic = checkCuBLASConfigDeterministic();
  if (C10_LIKELY(!deterministicAlgorithms() || cublas_config_deterministic)) {
    return;
  }

  auto msg = c10::str(
    "Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or ",
    "`at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because ",
    "it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this ",
    "case, you must set an environment variable before running your PyTorch application: ",
    cublas_config_var_name, "=", cublas_deterministic_configs[0], " or ",
    cublas_config_var_name, "=", cublas_deterministic_configs[1], ". For more information, go to ",
    "https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility"
  );

  if (deterministicAlgorithmsWarnOnly()) {
    TORCH_WARN(msg);
  } else {
    TORCH_CHECK(false, msg);
  }
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

bool Context::allowTF32CuBLAS() const {
#ifdef USE_ROCM
    const auto allow_tf32 = c10::utils::check_env(hipblaslt_allow_tf32);
    if (allow_tf32 != true) {
      return false;
    }
#endif
  return float32_matmul_precision != at::Float32MatmulPrecision::HIGHEST;
}

void Context::setAllowTF32CuBLAS(bool b) {
#ifdef USE_ROCM
  const auto allow_tf32 = c10::utils::check_env(hipblaslt_allow_tf32);
  if (allow_tf32 != true) {
    C10_LOG_FIRST_N(INFO, 10) << "torch.backends.cuda.matmul.allow_tf32 is not supported on ROCm by default. "
                              << "Please set environment variable HIPBLASLT_ALLOW_TF32=1 to enable it.";
    return;
  }
#endif
  float32_matmul_precision = b ? at::Float32MatmulPrecision::HIGH : at::Float32MatmulPrecision::HIGHEST;
}

Float32MatmulPrecision Context::float32MatmulPrecision() const {
  return float32_matmul_precision;
}

void Context::setFloat32MatmulPrecision(Float32MatmulPrecision p) {
  float32_matmul_precision = p;
}

void Context::setFloat32MatmulPrecision(const std::string &s) {
  auto match = [this](const std::string & s_) {
    // TODO: consider if CuDNN field needs to also be set for potential future CuDNN ops like multi-headed attention
    if (s_ == "highest") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGHEST;
      return true;
    } else if (s_ == "high") {
      float32_matmul_precision = at::Float32MatmulPrecision::HIGH;
      return true;
    } else if (s_ == "medium") {
      float32_matmul_precision = at::Float32MatmulPrecision::MEDIUM;
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
#ifdef USE_ROCM
    // AMD Instinct targets prefer hipblaslt
    static const bool hipblaslt_preferred = []() {
      static const std::vector<std::string> archs = {
          "gfx90a", "gfx942",
#if ROCM_VERSION >= 60500
          "gfx950"
#endif
      };
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
      static const std::vector<std::string> archs = {
          "gfx90a", "gfx942",
#if ROCM_VERSION >= 60300
          "gfx1100", "gfx1101", "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 60500
          "gfx950"
#endif
      };
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

void Context::setBlasPreferredBackend(at::BlasBackend b) {
#ifdef _MSC_VER
  TORCH_WARN_ONCE(
    "torch.backends.cuda.preferred_blas_library is an experimental feature. "
    "It is not supported on Windows."
  );
#else
  TORCH_CHECK((b != at::BlasBackend::Cublaslt) || hasCuBLASLt(),
      "Cannot set preferred backend to cuBLASLt if PyTorch has not been compiled with cuBLASLt.");
  TORCH_CHECK((b != at::BlasBackend::Ck) || hasROCM(),
      "Cannot set preferred backend to Ck if PyTorch has not been compiled for ROCm.");
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

at::ROCmFABackend Context::getROCmFAPreferredBackend() const {
  return rocm_fa_preferred_backend;
}

void Context::setROCmFAPreferredBackend(at::ROCmFABackend b) {

  // TODO: add plumbing for hasCK for validity checking
  TORCH_CHECK((b != at::ROCmFABackend::Ck) || hasROCM(),
      "Cannot set preferred flash attention backend to Ck if PyTorch has not been compiled for ROCm.");
#ifdef USE_ROCM
  if(b == at::ROCmFABackend::Ck) {
    static const bool ck_unsupported = []() {
      static const std::vector<std::string> archs = {
          "gfx90a",  "gfx942"
      };
      for (auto index: c10::irange(detail::getCUDAHooks().deviceCount())) {
        if (!detail::getCUDAHooks().isGPUArch(archs, index)) {
          TORCH_WARN_ONCE(
            "Attempting to use CK on an unsupported architecture! Cannot set backend to CK");
          return true;
        }
      }
      return false;
    }();
    if(!ck_unsupported) rocm_fa_preferred_backend = b;
  }
  else {
     rocm_fa_preferred_backend = b;
  }
#endif
  rocm_fa_preferred_backend = b;
}

bool Context::allowFP16ReductionCuBLAS() const {
  return allow_fp16_reduction_cublas;
}

void Context::setAllowFP16ReductionCuBLAS(bool b) {
  allow_fp16_reduction_cublas = b;
}

bool Context::allowBF16ReductionCuBLAS() const {
  return allow_bf16_reduction_cublas;
}

void Context::setAllowBF16ReductionCuBLAS(bool b) {
  allow_bf16_reduction_cublas = b;
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
  return quantized_engine.value_or(_quantized_engine);
}

void Context::setQEngine(at::QEngine e) {
  const auto& qengines = supportedQEngines();
  if (std::find(qengines.begin(), qengines.end(), e) != qengines.end()) {
    quantized_engine = e;
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
#ifdef C10_MOBILE
    engines.push_back(at::kNoQEngine);
#ifdef USE_PYTORCH_QNNPACK
    engines.push_back(at::kQNNPACK);
#endif
#else  // C10_MOBILE
#ifdef USE_PYTORCH_QNNPACK
    engines.push_back(at::kQNNPACK);
#endif
    engines.push_back(at::kNoQEngine);
#endif // C10_MOBILE

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
    if (true)
#endif
      TORCH_CHECK(false, "Float16 arithmetic is not supported by the CPU!");
  }
  allow_fp16_reduction_cpu = b;
}
} // namespace at
