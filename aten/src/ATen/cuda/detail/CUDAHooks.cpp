#include <ATen/cuda/detail/CUDAHooks.h>

#include <ATen/CUDAGenerator.h>
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <c10/util/Exception.h>

#include <THC/THC.h>
#include <THC/THCGeneral.hpp>

#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/cudnn-wrapper.h>
#endif

#ifdef USE_MAGMA
#include <magma.h>
#endif

#ifdef __HIP_PLATFORM_HCC__
#include <miopen/version.h>
#endif

#include <cuda.h>

#include <sstream>
#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace cuda {
namespace detail {

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
std::unique_ptr<THCState, void (*)(THCState*)> CUDAHooks::initCUDA() const {
  C10_LOG_API_USAGE_ONCE("aten.init.cuda");
  THCState* thc_state = THCState_alloc();

  THCudaInit(thc_state);
#ifdef USE_MAGMA
  THCMagma_init(thc_state);
#endif
  return std::unique_ptr<THCState, void (*)(THCState*)>(
      thc_state, [](THCState* p) {
        if (p)
          THCState_free(p);
      });
}

GeneratorImpl* CUDAHooks::getDefaultCUDAGenerator(DeviceIndex device_index) const {
  return at::cuda::detail::getDefaultCUDAGenerator(device_index);
}

Device CUDAHooks::getDeviceFromPtr(void* data) const {
  return at::cuda::getDeviceFromPtr(data);
}

bool CUDAHooks::isPinnedPtr(void* data) const {
  // First check if driver is broken/missing, in which case PyTorch CPU
  // functionalities should still work, we should report `false` here.
  if (!CUDAHooks::hasCUDA()) {
    return false;
  }
  // cudaPointerGetAttributes grabs context on the current device, so we set
  // device to one that already has context, if exists.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = CUDAHooks::getDevceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
  }
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, data);
#ifndef __HIP_PLATFORM_HCC__
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
#if CUDA_VERSION >= 10000
  return attr.type == cudaMemoryTypeHost;
#else
  return attr.memoryType == cudaMemoryTypeHost;
#endif
}

bool CUDAHooks::hasCUDA() const {
  return at::cuda::is_available();
}

bool CUDAHooks::hasMAGMA() const {
#ifdef USE_MAGMA
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::hasCuDNN() const {
  return AT_CUDNN_ENABLED();
}

#ifdef USE_DIRECT_NVRTC
static std::pair<std::unique_ptr<at::DynamicLibrary>, at::cuda::NVRTC*> load_nvrtc() {
  return std::make_pair(nullptr, at::cuda::load_nvrtc());
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

const at::cuda::NVRTC& CUDAHooks::nvrtc() const {
  // must hold onto DynamicLibrary otherwise it will unload
  static auto handle = load_nvrtc();
  return *handle.second;
}

int64_t CUDAHooks::current_device() const {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
  return -1;
}

bool CUDAHooks::hasPrimaryContext(int64_t device_index) const {
  TORCH_CHECK(device_index >= 0 && device_index < at::cuda::device_count(),
              "hasPrimaryContext expects a valid device index, but got device_index=", device_index);
  unsigned int ctx_flags;
  int ctx_is_active;
  AT_CUDA_DRIVER_CHECK(CUDAHooks::nvrtc().cuDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

c10::optional<int64_t> CUDAHooks::getDevceIndexWithPrimaryContext() const {
  // check current device first
  int64_t current_device_index = CUDAHooks::current_device();
  if (current_device_index >= 0) {
    if (CUDAHooks::hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (int64_t device_index = 0; device_index < CUDAHooks::getNumGPUs(); device_index++) {
    if (device_index == current_device_index) continue;
    if (CUDAHooks::hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return c10::nullopt;
}

Allocator* CUDAHooks::getPinnedMemoryAllocator() const {
  return at::cuda::getPinnedMemoryAllocator();
}

bool CUDAHooks::compiledWithCuDNN() const {
  return AT_CUDNN_ENABLED();
}

bool CUDAHooks::compiledWithMIOpen() const {
  return AT_ROCM_ENABLED();
}

bool CUDAHooks::supportsDilatedConvolutionWithCuDNN() const {
#if AT_CUDNN_ENABLED()
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
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

long CUDAHooks::versionCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_VERSION;
#else
  AT_ERROR("Cannot query CuDNN version if ATen_cuda is not built with CuDNN");
#endif
}

std::string CUDAHooks::showConfig() const {
  std::ostringstream oss;

  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);

  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };

#ifndef __HIP_PLATFORM_HCC__
  oss << "  - CUDA Runtime ";
#else
  oss << "  - HIP Runtime ";
#endif
  printCudaStyleVersion(runtimeVersion);
  oss << "\n";

  // TODO: Make HIPIFY understand CUDART_VERSION macro
#ifndef __HIP_PLATFORM_HCC__
  if (runtimeVersion != CUDART_VERSION) {
    oss << "  - Built with CUDA Runtime ";
    printCudaStyleVersion(CUDART_VERSION);
    oss << "\n";
  }
  oss << "  - NVCC architecture flags: " << NVCC_FLAGS_EXTRA << "\n";
#endif

#ifndef __HIP_PLATFORM_HCC__
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

#ifdef USE_MAGMA
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
#ifndef __HIP_PLATFORM_HCC__
  return at::native::detail::cufft_get_plan_cache_max_size_impl(device_index);
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

void CUDAHooks::cuFFTSetPlanCacheMaxSize(int64_t device_index, int64_t max_size) const {
#ifndef __HIP_PLATFORM_HCC__
  at::native::detail::cufft_set_plan_cache_max_size_impl(device_index, max_size);
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

int64_t CUDAHooks::cuFFTGetPlanCacheSize(int64_t device_index) const {
#ifndef __HIP_PLATFORM_HCC__
  return at::native::detail::cufft_get_plan_cache_size_impl(device_index);
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

void CUDAHooks::cuFFTClearPlanCache(int64_t device_index) const {
#ifndef __HIP_PLATFORM_HCC__
  at::native::detail::cufft_clear_plan_cache_impl(device_index);
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

int CUDAHooks::getNumGPUs() const {
  return at::cuda::device_count();
}

// Sigh, the registry doesn't support namespaces :(
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks);

} // namespace detail
} // namespace cuda
} // namespace at
