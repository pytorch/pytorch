#include "ATen/Config.h"

#include "Context.h"

#include <thread>
#include <mutex>
#include <sstream>
#include <string>
#include <stdexcept>

#if AT_CUDA_ENABLED()
#include <cuda.h>
#include "THC/THC.h"
#include "ATen/CUDAGenerator.h"
#endif
#include "ATen/CPUGenerator.h"

#ifdef USE_SSE3
#include <pmmintrin.h>
#endif

namespace at {

static inline void errorHandler(const char * msg, void * data) {
  throw std::runtime_error(msg);
}
static inline void argErrorHandler(int arg, const char * msg, void * data) {
  std::stringstream new_error;
  new_error << "invalid argument " << arg << ": " << msg;
  throw std::runtime_error(new_error.str());
}

Context::Context()
: thc_state(nullptr) {

  THSetDefaultErrorHandler(errorHandler,nullptr);
  THSetDefaultArgErrorHandler(argErrorHandler,nullptr);

  generator_registry[static_cast<int>(Backend::CPU)]
    .reset(new CPUGenerator(this));
  Type::registerAll(this);
}
void Context::doInitCUDA() {
#if AT_CUDA_ENABLED()
  thc_state = THCState_alloc();
  THCState_setDeviceAllocator(thc_state, THCCachingAllocator_get());
  thc_state->cudaHostAllocator = &THCCachingHostAllocator;
  THCudaInit(thc_state);
  generator_registry[static_cast<int>(Backend::CUDA)]
    .reset(new CUDAGenerator(this));
#endif
}
Context::~Context() {
#if AT_CUDA_ENABLED()
  if(thc_state)
    THCState_free(thc_state);
#endif
}

Context & globalContext() {
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

bool Context::deterministicCuDNN() const {
  return deterministic_cudnn;
}

void Context::setDeterministicCuDNN(bool b) {
  deterministic_cudnn = b;
}

bool Context::benchmarkCuDNN() const {
  return benchmark_cudnn;
}

void Context::setBenchmarkCuDNN(bool b) {
  benchmark_cudnn = b;
}

bool Context::hasMKL() const {
#if AT_MKL_ENABLED()
  return true;
#else
  return false;
#endif
}

bool Context::hasCUDA() const {
#if AT_CUDA_ENABLED()
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver) {
    return false;
  }
  return true;
#else
  return false;
#endif
}

#if AT_CUDA_ENABLED()
cudaStream_t Context::getCurrentCUDAStream() const {
  return THCState_getCurrentStream(thc_state);
}
struct cudaDeviceProp* Context::getCurrentDeviceProperties() const {
  return THCState_getCurrentDeviceProperties(thc_state);
}
struct cudaDeviceProp* Context::getDeviceProperties(int device) const {
  return THCState_getDeviceProperties(thc_state, device);
}
#else
cudaStream_t Context::getCurrentCUDAStream() const {
  throw std::runtime_error("ATen not compiled with CUDA");
}
struct cudaDeviceProp* Context::getCurrentDeviceProperties() const {
  throw std::runtime_error("ATen not compiled with CUDA");
}
struct cudaDeviceProp* Context::getDeviceProperties(int device) const {
  throw std::runtime_error("ATen not compiled with CUDA");
}
#endif

int64_t Context::current_device() const {
#if AT_CUDA_ENABLED()
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
#endif
  return -1;
}

bool Context::setFlushDenormal(bool on) {
#ifdef USE_SSE3
  // Setting flush-to-zero (FTZ) flag
  _MM_SET_FLUSH_ZERO_MODE(on ? _MM_FLUSH_ZERO_ON
                             : _MM_FLUSH_ZERO_OFF);

  // Setting denormals-are-zero (DAZ) flag
  _MM_SET_DENORMALS_ZERO_MODE(on ? _MM_DENORMALS_ZERO_ON
                                 : _MM_DENORMALS_ZERO_OFF);
  return true;
#else
  return false;
#endif
}

}
