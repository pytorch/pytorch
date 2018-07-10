#include <ATen/cuda/detail/CUDAHooks.h>

#include <ATen/CUDAGenerator.h>
#include <ATen/Context.h>
#include <ATen/Error.h>
#include <ATen/RegisterCUDA.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/detail/CUDAHooksInterface.h>

#include "THC/THC.h"
#include <THC/THCGeneral.hpp>

#if AT_CUDNN_ENABLED()
#include "ATen/cudnn/cudnn-wrapper.h"
#endif

#include <cuda.h>

#include <cstddef>
#include <functional>
#include <memory>

namespace at {
namespace cuda {
namespace detail {
namespace {

void check_status(int32_t status) {
  AT_CHECK(
      static_cast<cudaError_t>(status) == cudaSuccess,
      "CUDA error (",
      static_cast<int32_t>(status),
      "): ",
      cudaGetErrorString(static_cast<cudaError_t>(status)));
}

void set_device(int32_t device) {
  check_status(cudaSetDevice(device));
}

void get_device(int32_t* device) {
  check_status(cudaGetDevice(device));
}

void unchecked_set_device(int32_t device) {
  const auto return_code = cudaSetDevice(device);
  (void)return_code;
}

void cuda_stream_create_with_priority(
  cudaStream_t* pStream
, int32_t flags
, int32_t priority) {
#ifndef __HIP_PLATFORM_HCC__
  check_status(cudaStreamCreateWithPriority(pStream, flags, priority));
#else
  check_status(cudaStreamCreateWithFlags(pStream, flags));
#endif
}

void cuda_stream_destroy(cudaStream_t stream) {
  check_status(cudaStreamDestroy(stream));
}

struct DynamicCUDAInterfaceSetter {
  DynamicCUDAInterfaceSetter() {
    at::detail::DynamicCUDAInterface::set_device = set_device;
    at::detail::DynamicCUDAInterface::get_device = get_device;
    at::detail::DynamicCUDAInterface::unchecked_set_device =
        unchecked_set_device;
    at::detail::DynamicCUDAInterface::cuda_stream_create_with_priority = 
      cuda_stream_create_with_priority;
    at::detail::DynamicCUDAInterface::cuda_stream_destroy = cuda_stream_destroy;
  }
};

// Single, global, static (because of the anonymous namespace) instance, whose
// constructor will set the static members of `DynamicCUDAInterface` to CUDA
// functions when the ATen CUDA library is loaded.
DynamicCUDAInterfaceSetter _;
} // namespace

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
std::unique_ptr<THCState, void (*)(THCState*)> CUDAHooks::initCUDA() const {
  THCState* thc_state = THCState_alloc();
  THCState_setDeviceAllocator(thc_state, THCCachingAllocator_get());
  thc_state->cudaHostAllocator = &THCCachingHostAllocator;
  THCudaInit(thc_state);
  return std::unique_ptr<THCState, void (*)(THCState*)>(
      thc_state, [](THCState* p) {
        if (p)
          THCState_free(p);
      });
}

std::unique_ptr<Generator> CUDAHooks::initCUDAGenerator(
    Context* context) const {
  return std::unique_ptr<Generator>(new CUDAGenerator(context));
}

bool CUDAHooks::hasCUDA() const {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver) {
    return false;
  }
  return true;
}

bool CUDAHooks::hasCuDNN() const {
  return AT_CUDNN_ENABLED();
}

#ifndef __HIP_PLATFORM_HCC__
cusparseHandle_t CUDAHooks::getCurrentCUDASparseHandle(THCState* thc_state) const {
  return THCState_getCurrentSparseHandle(thc_state);
}
#endif
struct cudaDeviceProp* CUDAHooks::getCurrentDeviceProperties(
    THCState* thc_state) const {
  return THCState_getCurrentDeviceProperties(thc_state);
}
struct cudaDeviceProp* CUDAHooks::getDeviceProperties(
    THCState* thc_state,
    int device) const {
  return THCState_getDeviceProperties(thc_state, device);
}

int64_t CUDAHooks::current_device() const {
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
  return -1;
}

Allocator* CUDAHooks::getPinnedMemoryAllocator() const {
  return at::cuda::getPinnedMemoryAllocator();
}

void CUDAHooks::registerCUDATypes(Context* context) const {
  register_cuda_types(context);
}

bool CUDAHooks::compiledWithCuDNN() const {
  return AT_CUDNN_ENABLED();
}

bool CUDAHooks::supportsDilatedConvolutionWithCuDNN() const {
#if AT_CUDNN_ENABLED()
  cudaDeviceProp* prop =
      getCurrentDeviceProperties(globalContext().getTHCState());
  // NOTE: extra parenthesis around numbers disable clang warnings about
  // dead code
  return (
      (CUDNN_VERSION >= (6021)) ||
      (CUDNN_VERSION >= (6000) && prop->major >= 5));
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

double CUDAHooks::batchnormMinEpsilonCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_BN_MIN_EPSILON;
#else
  AT_ERROR(
      "Cannot query CUDNN_BN_MIN_EPSILON if ATen_cuda is not built with CuDNN");
#endif
}

int64_t CUDAHooks::cuFFTGetPlanCacheMaxSize() const {
#ifndef __HIP_PLATFORM_HCC__
  return at::native::detail::cufft_get_plan_cache_max_size_impl();
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

void CUDAHooks::cuFFTSetPlanCacheMaxSize(int64_t max_size) const {
#ifndef __HIP_PLATFORM_HCC__
  at::native::detail::cufft_set_plan_cache_max_size_impl(max_size);
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

int64_t CUDAHooks::cuFFTGetPlanCacheSize() const {
#ifndef __HIP_PLATFORM_HCC__
  return at::native::detail::cufft_get_plan_cache_size_impl();
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

void CUDAHooks::cuFFTClearPlanCache() const {
#ifndef __HIP_PLATFORM_HCC__
  at::native::detail::cufft_clear_plan_cache_impl();
#else
  AT_ERROR("cuFFT with HIP is not supported");
#endif
}

int CUDAHooks::getNumGPUs() const {
  int count;
  auto err = cudaGetDeviceCount(&count);
  if (err == cudaErrorNoDevice) {
    return 0;
  } else if (err != cudaSuccess) {
    AT_ERROR(
        "CUDA error (", static_cast<int>(err), "): ", cudaGetErrorString(err));
  }
  return count;
}

// Sigh, the registry doesn't support namespaces :(
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks);

} // namespace detail
} // namespace cuda
} // namespace at
