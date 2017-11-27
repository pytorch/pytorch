#include "ATen/Config.h"

#include "Context.h"

#include <thread>
#include <mutex>
#include <sstream>

#if AT_CUDA_ENABLED()
#include "THC/THC.h"
#include "ATen/CUDAGenerator.h"
#endif
#include "ATen/CPUGenerator.h"

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
#endif

}
