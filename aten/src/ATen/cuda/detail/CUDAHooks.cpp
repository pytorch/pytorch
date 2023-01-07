#include <ATen/cuda/detail/CUDAHooks.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/core/Vitals.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/irange.h>

#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/cudnn-wrapper.h>
#endif

#if AT_MAGMA_ENABLED()
#include <magma_v2.h>
#endif

#if defined(USE_ROCM)
#include <miopen/version.h>
#endif

#ifndef USE_ROCM
#include <ATen/cuda/detail/LazyNVRTC.h>
#endif

#include <cuda.h>

#include <sstream>
#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace cuda {
namespace detail {

const at::cuda::NVRTC& nvrtc();
int64_t current_device();

static void (*magma_init_fn)() = nullptr;

void set_magma_init_fn(void (*fn)()) {
  magma_init_fn = fn;
}

// Sets the CUDA_MODULE_LOADING environment variable
// if it's not set by the user.
void maybe_set_cuda_module_loading(const std::string &def_value) {
  auto value = std::getenv("CUDA_MODULE_LOADING");
  if (!value) {
#ifdef _WIN32
    auto env_var = "CUDA_MODULE_LOADING=" + def_value;
    _putenv(env_var.c_str());
#else
    setenv("CUDA_MODULE_LOADING", def_value.c_str(), 1);
#endif
  }
}

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
void CUDAHooks::initCUDA() const {
  C10_LOG_API_USAGE_ONCE("aten.init.cuda");
  // Force the update to enable unit testing. This code get executed before unit tests
  // have a chance to enable vitals.
  at::vitals::VitalsAPI.setVital("CUDA", "used", "true", /* force = */ true);

  maybe_set_cuda_module_loading("LAZY");
  const auto num_devices = c10::cuda::device_count_ensure_non_zero();
  c10::cuda::CUDACachingAllocator::init(num_devices);
  at::cuda::detail::init_p2p_access_cache(num_devices);

#if AT_MAGMA_ENABLED()
  TORCH_INTERNAL_ASSERT(magma_init_fn != nullptr, "Cannot initialize magma, init routine not set");
  magma_init_fn();
#endif
}

const Generator& CUDAHooks::getDefaultCUDAGenerator(DeviceIndex device_index) const {
  return at::cuda::detail::getDefaultCUDAGenerator(device_index);
}

Device CUDAHooks::getDeviceFromPtr(void* data) const {
  return at::cuda::getDeviceFromPtr(data);
}

bool CUDAHooks::isPinnedPtr(void* data) const {
  // First check if driver is broken/missing, in which case PyTorch CPU
  // functionalities should still work, we should report `false` here.
  if (!at::cuda::is_available()) {
    return false;
  }
  // cudaPointerGetAttributes grabs context on the current device, so we set
  // device to one that already has context, if exists.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = getDeviceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
  }
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, data);
#if !defined(USE_ROCM)
  if (err == cudaErrorInvalidValue) {
    cudaGetLastError();
    return false;
  }
  AT_CUDA_CHECK(err);
#else
  // HIP throws hipErrorUnknown here
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
#endif
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10000
  return attr.type == cudaMemoryTypeHost;
#else
  return attr.memoryType == cudaMemoryTypeHost;
#endif
}

bool CUDAHooks::hasCUDA() const {
  return at::cuda::is_available();
}

bool CUDAHooks::hasMAGMA() const {
#if AT_MAGMA_ENABLED()
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::hasCuDNN() const {
  return AT_CUDNN_ENABLED();
}

bool CUDAHooks::hasCuSOLVER() const {
#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION)
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::hasROCM() const {
  // Currently, this is same as `compiledWithMIOpen`.
  // But in future if there are ROCm builds without MIOpen,
  // then `hasROCM` should return true while `compiledWithMIOpen`
  // should return false
  return AT_ROCM_ENABLED();
}

#if defined(USE_DIRECT_NVRTC)
static std::pair<std::unique_ptr<at::DynamicLibrary>, at::cuda::NVRTC*> load_nvrtc() {
  return std::make_pair(nullptr, at::cuda::load_nvrtc());
}
#elif !defined(USE_ROCM)
static std::pair<std::unique_ptr<at::DynamicLibrary>, at::cuda::NVRTC*> load_nvrtc() {
  return std::make_pair(nullptr, &at::cuda::detail::lazyNVRTC);
}
#else
static std::pair<std::unique_ptr<at::DynamicLibrary>, at::cuda::NVRTC*> load_nvrtc() {
#if defined(_WIN32)
  std::string libcaffe2_nvrtc = "caffe2_nvrtc.dll";
#elif defined(__APPLE__)
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.dylib";
#else
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.so";
#endif
  std::unique_ptr<at::DynamicLibrary> libnvrtc_stub(
      new at::DynamicLibrary(libcaffe2_nvrtc.c_str()));
  auto fn = (at::cuda::NVRTC * (*)()) libnvrtc_stub->sym("load_nvrtc");
  return std::make_pair(std::move(libnvrtc_stub), fn());
}
#endif

const at::cuda::NVRTC& nvrtc() {
  // must hold onto DynamicLibrary otherwise it will unload
  static auto handle = load_nvrtc();
  return *handle.second;
}

const at::cuda::NVRTC& CUDAHooks::nvrtc() const {
  return at::cuda::detail::nvrtc();
}

int64_t current_device() {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
  return -1;
}

int64_t CUDAHooks::current_device() const {
  return at::cuda::detail::current_device();
}

bool hasPrimaryContext(int64_t device_index) {
  TORCH_CHECK(device_index >= 0 && device_index < at::cuda::device_count(),
              "hasPrimaryContext expects a valid device index, but got device_index=", device_index);
  unsigned int ctx_flags;
  // In standalone tests of cuDevicePrimaryCtxGetState, I've seen the "active" argument end up with weird
  // (garbage-looking nonzero) values when the context is not active, unless I initialize it to zero.
  int ctx_is_active = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

bool CUDAHooks::hasPrimaryContext(int64_t device_index) const {
  return at::cuda::detail::hasPrimaryContext(device_index);
}

c10::optional<int64_t> getDeviceIndexWithPrimaryContext() {
  // check current device first
  int64_t current_device_index = current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(at::cuda::device_count())) {
    if (device_index == current_device_index) continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return c10::nullopt;
}

Allocator* CUDAHooks::getPinnedMemoryAllocator() const {
  return at::cuda::getPinnedMemoryAllocator();
}

Allocator* CUDAHooks::getCUDADeviceAllocator() const {
  return at::cuda::getCUDADeviceAllocator();
}

bool CUDAHooks::compiledWithCuDNN() const {
  return AT_CUDNN_ENABLED();
}

bool CUDAHooks::compiledWithMIOpen() const {
  return AT_ROCM_ENABLED();
}

bool CUDAHooks::supportsDilatedConvolutionWithCuDNN() const {
#if AT_CUDNN_ENABLED()
  // NOTE: extra parenthesis around numbers disable clang warnings about
  // dead code
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::supportsDepthwiseConvolutionWithCuDNN() const {
#if AT_CUDNN_ENABLED()
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // Check for Volta cores
  if (prop->major >= 7) {
    return true;
  } else {
    return false;
  }
#else
  return false;
#endif
}

bool CUDAHooks::supportsBFloat16ConvolutionWithCuDNNv8() const {
#if AT_CUDNN_ENABLED()
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // Check for Volta cores
  if (prop->major >= 8) {
    return true;
  } else {
    return false;
  }
#else
  return false;
#endif
}

long CUDAHooks::versionCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_VERSION;
#else
  AT_ERROR("Cannot query CuDNN version if ATen_cuda is not built with CuDNN");
#endif
}

long CUDAHooks::versionCUDART() const {
#ifdef CUDART_VERSION
  return CUDART_VERSION;
#else
  TORCH_CHECK(
    false,
    "Cannot query CUDART version because CUDART is not available");
#endif
}

bool CUDAHooks::hasCUDART() const {
#ifdef CUDART_VERSION
  return true;
#else
  return false;
#endif
}

std::string CUDAHooks::showConfig() const {
  std::ostringstream oss;

  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);

  auto printCudaStyleVersion = [&](int v) {
#ifdef USE_ROCM
    // HIP_VERSION value format was changed after ROCm v4.2 to include the patch number
    if(v < 500) {
      // If major=xx, minor=yy then format -> xxyy
      oss << (v / 100) << "." << (v % 10);
    }
    else {
      // If major=xx, minor=yy & patch=zzzzz then format -> xxyyzzzzz
      oss << (v / 10000000) << "." << (v / 100000 % 100) << "." << (v % 100000);
    }
#else
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
#endif
  };

#if !defined(USE_ROCM)
  oss << "  - CUDA Runtime ";
#else
  oss << "  - HIP Runtime ";
#endif
  printCudaStyleVersion(runtimeVersion);
  oss << "\n";

  // TODO: Make HIPIFY understand CUDART_VERSION macro
#if !defined(USE_ROCM)
  if (runtimeVersion != CUDART_VERSION) {
    oss << "  - Built with CUDA Runtime ";
    printCudaStyleVersion(CUDART_VERSION);
    oss << "\n";
  }
  oss << "  - NVCC architecture flags: " << NVCC_FLAGS_EXTRA << "\n";
#endif

#if !defined(USE_ROCM)
#if AT_CUDNN_ENABLED()


  auto printCudnnStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 100 % 10);
    if (v % 100 != 0) {
      oss << "." << (v % 100);
    }
  };

  size_t cudnnVersion = cudnnGetVersion();
  oss << "  - CuDNN ";
  printCudnnStyleVersion(cudnnVersion);
  size_t cudnnCudartVersion = cudnnGetCudartVersion();
  if (cudnnCudartVersion != CUDART_VERSION) {
    oss << "  (built against CUDA ";
    printCudaStyleVersion(cudnnCudartVersion);
    oss << ")";
  }
  oss << "\n";
  if (cudnnVersion != CUDNN_VERSION) {
    oss << "    - Built with CuDNN ";
    printCudnnStyleVersion(CUDNN_VERSION);
    oss << "\n";
  }
#endif
#else
  // TODO: Check if miopen has the functions above and unify
  oss << "  - MIOpen " << MIOPEN_VERSION_MAJOR << "." << MIOPEN_VERSION_MINOR << "." << MIOPEN_VERSION_PATCH << "\n";
#endif

#if AT_MAGMA_ENABLED()
  oss << "  - Magma " << MAGMA_VERSION_MAJOR << "." << MAGMA_VERSION_MINOR << "." << MAGMA_VERSION_MICRO << "\n";
#endif

  return oss.str();
}

double CUDAHooks::batchnormMinEpsilonCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_BN_MIN_EPSILON;
#else
  AT_ERROR(
      "Cannot query CUDNN_BN_MIN_EPSILON if ATen_cuda is not built with CuDNN");
#endif
}

int64_t CUDAHooks::cuFFTGetPlanCacheMaxSize(int64_t device_index) const {
  return at::native::detail::cufft_get_plan_cache_max_size_impl(device_index);
}

void CUDAHooks::cuFFTSetPlanCacheMaxSize(int64_t device_index, int64_t max_size) const {
  at::native::detail::cufft_set_plan_cache_max_size_impl(device_index, max_size);
}

int64_t CUDAHooks::cuFFTGetPlanCacheSize(int64_t device_index) const {
  return at::native::detail::cufft_get_plan_cache_size_impl(device_index);
}

void CUDAHooks::cuFFTClearPlanCache(int64_t device_index) const {
  at::native::detail::cufft_clear_plan_cache_impl(device_index);
}

int CUDAHooks::getNumGPUs() const {
  return at::cuda::device_count();
}

void CUDAHooks::deviceSynchronize(int64_t device_index) const {
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  c10::cuda::device_synchronize();
}

// Sigh, the registry doesn't support namespaces :(
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks);

} // namespace detail
} // namespace cuda
} // namespace at
