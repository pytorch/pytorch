#include <ATen/Config.h>

#include <ATen/Context.h>

#include <c10/core/TensorOptions.h>
#include <c10/core/CPUAllocator.h>

#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <ATen/Tensor.h>
#include <ATen/cpu/FlushDenormal.h>

#include <TH/TH.h> // for USE_LAPACK

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#endif // USE_FBGEMM

namespace at {

Context::Context()
    : thc_state(nullptr, [](THCState* p) { /* no-op */ }),
      thh_state(nullptr, [](THHState* p) { /* no-op */ }) {}

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

bool Context::deterministicAlgorithms() const {
  return _deterministic_algorithms;
}

void Context::setDeterministicAlgorithms(bool b) {
  if (b) {
    TORCH_WARN_ONCE("torch.use_deterministic_algorithms is in beta, and its design and"
      " functionality may change in the future.");
  }

  _deterministic_algorithms = b;
}

void Context::alertNotDeterministic(c10::string_view const& caller) {
  if (globalContext().deterministicAlgorithms()) {
    TORCH_CHECK(false,
      caller, " does not have a deterministic implementation, but you set "
      "'torch.use_deterministic_algorithms(True)'. You can turn off determinism ",
      "just for this operation if that's acceptable for your application. You "
      "can also file an issue at https://github.com/pytorch/pytorch/issues "
      "to help us prioritize adding deterministic support for this operation.");
  }
}

bool Context::allowTF32CuDNN() const {
  return allow_tf32_cudnn;
}

void Context::setAllowTF32CuDNN(bool b) {
  allow_tf32_cudnn = b;
}

static const char cublas_config_var_name[] = "CUBLAS_WORKSPACE_CONFIG";
static const char* const cublas_deterministic_configs[] = { ":4096:8", ":16:8" };

bool Context::checkCuBLASConfigDeterministic() {
  bool cublas_config_deterministic = true;
  // If using CUDA 10.2 or greater, need to make sure CuBLAS workspace config
  // is set to deterministic setting
  if (hasCUDART() && (versionCUDART() >= 10020)) {
    char* workspace_config = std::getenv(cublas_config_var_name);
    cublas_config_deterministic = (workspace_config != nullptr) && (
      (strcmp(workspace_config, cublas_deterministic_configs[0]) == 0)
      || (strcmp(workspace_config, cublas_deterministic_configs[1]) == 0)
    );
  }
  return cublas_config_deterministic;
}

void Context::alertCuBLASConfigNotDeterministic() {
  static bool cublas_config_deterministic = checkCuBLASConfigDeterministic();
  TORCH_CHECK(!deterministicAlgorithms() || cublas_config_deterministic,
    "Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or ",
    "`at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because ",
    "it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this ",
    "case, you must set an environment variable before running your PyTorch application: ",
    cublas_config_var_name, "=", cublas_deterministic_configs[0], " or ",
    cublas_config_var_name, "=", cublas_deterministic_configs[1], ". For more information, go to ",
    "https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility"
  );
}

bool Context::benchmarkCuDNN() const {
  return benchmark_cudnn;
}

void Context::setBenchmarkCuDNN(bool b) {
  benchmark_cudnn = b;
}

bool Context::allowTF32CuBLAS() const {
  return allow_tf32_cublas;
}

void Context::setAllowTF32CuBLAS(bool b) {
  allow_tf32_cublas = b;
}

bool Context::hasMKL() const {
#if AT_MKL_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasMKLDNN() const {
#if AT_MKLDNN_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasOpenMP() const {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

bool Context::hasLAPACK() const {
#ifdef USE_LAPACK
  return true;
#else
  return false;
#endif
}

at::QEngine Context::qEngine() const {
  // If wasn't explicitly set - take the last one available
  return quantized_engine.value_or(supportedQEngines().back());
}

void Context::setQEngine(at::QEngine e) {
  const auto& qengines = supportedQEngines();
  if (std::find(qengines.begin(), qengines.end(), e) != qengines.end()) {
    quantized_engine = e;
    return;
  }
  TORCH_CHECK(false, "quantized engine ", toString(e), " is not supported");
}

const std::vector<at::QEngine>& Context::supportedQEngines() const {
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

#ifdef USE_FBGEMM
    if (fbgemm::fbgemmSupportedCPU()) {
      engines.push_back(at::kFBGEMM);
    }
#endif
    return engines;
  }();
  return supported_qengines;
}

bool Context::isXNNPACKAvailable() const {
#ifdef USE_XNNPACK
  return true;
#else
  return false;
#endif
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
//    means the allow_tf32 flags are overrided and tf32 is force disabled
// override_allow_tf32_flag = false
//    means the original allow_tf32 flags are followed
thread_local bool override_allow_tf32_flag = false;

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

bool Context::areVmapFallbackWarningsEnabled() const {
  return display_vmap_fallback_warnings_;
}

void Context::setDisplayVmapFallbackWarnings(bool enabled) {
  display_vmap_fallback_warnings_ = enabled;
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
} // namespace at
