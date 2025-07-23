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
#include <c10/util/env.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

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
#include <memory>

namespace c10::cuda::_internal {
void setHasPrimaryContext(bool (*func)(DeviceIndex));
}

namespace at::cuda::detail {

const at::cuda::NVRTC& nvrtc();
DeviceIndex current_device();

static void (*magma_init_fn)() = nullptr;

void set_magma_init_fn(void (*fn)()) {
  magma_init_fn = fn;
}

namespace {
bool _hasPrimaryContext(DeviceIndex device_index) {
  TORCH_CHECK(device_index >= 0 && device_index < at::cuda::device_count(),
              "hasPrimaryContext expects a valid device index, but got device_index=", device_index);
  unsigned int ctx_flags = 0;
  // In standalone tests of cuDevicePrimaryCtxGetState, I've seen the "active" argument end up with weird
  // (garbage-looking nonzero) values when the context is not active, unless I initialize it to zero.
  int ctx_is_active = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

// Register hasPrimaryContext back to c10::cuda
struct _Initializer {
  _Initializer() {
      c10::cuda::_internal::setHasPrimaryContext(_hasPrimaryContext);
  }
  ~_Initializer() {
      c10::cuda::_internal::setHasPrimaryContext(nullptr);
  }
} initializer;
} // anonymous namespace


// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
void CUDAHooks::init() const {
  C10_LOG_API_USAGE_ONCE("aten.init.cuda");
  // Force the update to enable unit testing. This code get executed before unit tests
  // have a chance to enable vitals.
  at::vitals::VitalsAPI.setVital("CUDA", "used", "true", /* force = */ true);

  // Sets the CUDA_MODULE_LOADING environment variable
  // if it's not set by the user.
  // CUDA_MODULE_LOADING="LAZY" is default for all drivers released for CUDA 12.2+.
  // Check the driver version and only set the env variable if needed.
  bool set_lazy_module_loading = true;
  #if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
  auto driver_api = c10::cuda::DriverAPI::get();
  // Initialize NVML
  if (driver_api->nvmlInit_v2_() == NVML_SUCCESS) {
    // Get the driver version
    int version = -1;
    auto res = driver_api->nvmlSystemGetCudaDriverVersion_v2_(&version);
    if (res == NVML_SUCCESS) {
      // Check if driver is sufficiently new
      if (version >= 12020) {
        set_lazy_module_loading = false;
      }
    }
  }
  #endif
  if (set_lazy_module_loading) {
    c10::utils::set_env("CUDA_MODULE_LOADING", "LAZY", false);
  }
  const auto num_devices = c10::cuda::device_count_ensure_non_zero();
  c10::cuda::CUDACachingAllocator::init(num_devices);
  at::cuda::detail::init_p2p_access_cache(num_devices);

#if AT_MAGMA_ENABLED()
  TORCH_INTERNAL_ASSERT(magma_init_fn != nullptr, "Cannot initialize magma, init routine not set");
  magma_init_fn();
#endif
}

const Generator& CUDAHooks::getDefaultGenerator(DeviceIndex device_index) const {
  return at::cuda::detail::getDefaultCUDAGenerator(device_index);
}

Generator CUDAHooks::getNewGenerator(DeviceIndex device_index) const {
  return make_generator<at::CUDAGeneratorImpl>(device_index);
}

Device CUDAHooks::getDeviceFromPtr(void* data) const {
  return at::cuda::getDeviceFromPtr(data);
}

bool CUDAHooks::isPinnedPtr(const void* data) const {
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
  cudaPointerAttributes attr{};
  // We do not believe that CUDA needs mutable access to the data
  // here.
  cudaError_t err = cudaPointerGetAttributes(&attr, data);
#if !defined(USE_ROCM)
  if (err == cudaErrorInvalidValue) {
    (void)cudaGetLastError(); // clear CUDA error
    return false;
  }
  AT_CUDA_CHECK(err);
#else
  // HIP throws hipErrorUnknown here
  if (err != cudaSuccess) {
    (void)cudaGetLastError(); // clear HIP error
    return false;
  }
#endif
  return attr.type == cudaMemoryTypeHost;
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
#elif AT_ROCM_ENABLED()
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::hasCuBLASLt() const {
#if defined(CUDART_VERSION)
  return true;
#elif AT_ROCM_ENABLED()
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

DeviceIndex current_device() {
  c10::DeviceIndex device = 0;
  cudaError_t err = c10::cuda::GetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
  return -1;
}

/**
 * DEPRECATED: use getCurrentDevice() instead
 */
DeviceIndex CUDAHooks::current_device() const {
  return at::cuda::detail::current_device();
}

bool CUDAHooks::hasPrimaryContext(DeviceIndex device_index) const {
  return _hasPrimaryContext(device_index);
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
  TORCH_CHECK(false, "Cannot query CuDNN version if ATen_cuda is not built with CuDNN");
#endif
}

long CUDAHooks::versionMIOpen() const {
#if AT_ROCM_ENABLED()
  return MIOPEN_VERSION_MAJOR * 10000 +
         MIOPEN_VERSION_MINOR * 100 +
         MIOPEN_VERSION_PATCH;
#else
  TORCH_CHECK(false, "Cannot query MIOpen version if ATen_cuda is not built with ROCm");
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

  int runtimeVersion = 0;
  AT_CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

  auto printCudaStyleVersion = [&](size_t v) {
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


  auto printCudnnStyleVersion = [&](size_t v) {
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
  TORCH_CHECK(false,
      "Cannot query CUDNN_BN_MIN_EPSILON if ATen_cuda is not built with CuDNN");
#endif
}

int64_t CUDAHooks::cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const {
  return at::native::detail::cufft_get_plan_cache_max_size_impl(device_index);
}

void CUDAHooks::cuFFTSetPlanCacheMaxSize(DeviceIndex device_index, int64_t max_size) const {
  at::native::detail::cufft_set_plan_cache_max_size_impl(device_index, max_size);
}

int64_t CUDAHooks::cuFFTGetPlanCacheSize(DeviceIndex device_index) const {
  return at::native::detail::cufft_get_plan_cache_size_impl(device_index);
}

void CUDAHooks::cuFFTClearPlanCache(DeviceIndex device_index) const {
  at::native::detail::cufft_clear_plan_cache_impl(device_index);
}

/**
 * DEPRECATED: use deviceCount() instead
 */
int CUDAHooks::getNumGPUs() const {
  return at::cuda::device_count();
}

DeviceIndex CUDAHooks::deviceCount() const {
  return at::cuda::device_count();
}

DeviceIndex CUDAHooks::getCurrentDevice() const {
  return at::cuda::detail::current_device();
}

#ifdef USE_ROCM
bool CUDAHooks::isGPUArch(const std::vector<std::string>& archs, DeviceIndex device_index) const {
  hipDeviceProp_t* prop;
  if (device_index == -1){
      prop = at::cuda::getCurrentDeviceProperties();
  } else {
      prop = at::cuda::getDeviceProperties(device_index);
  }

  std::string device_arch = prop->gcnArchName;
  for (std::string arch : archs) {
      size_t substring = device_arch.find(arch);
      if (substring != std::string::npos) {
          return true;
      }
  }
  return false;
}
#endif

void CUDAHooks::deviceSynchronize(DeviceIndex device_index) const {
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  c10::cuda::device_synchronize();
}

// Sigh, the registry doesn't support namespaces :(
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks)

} // namespace at::cuda::detail
