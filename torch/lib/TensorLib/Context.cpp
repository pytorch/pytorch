#include "Context.h"
#include <thread>
#include <mutex>

#ifdef TENSORLIB_CUDA_ENABLED
#include "THC/THC.h"
#include "TensorLib/CUDAGenerator.h"
#endif
#include "TensorLib/CPUGenerator.h"

namespace tlib {

static inline void errorHandler(const char * msg, void * data) {
  throw std::runtime_error(msg);
}

Context::Context() {

  THSetDefaultErrorHandler(errorHandler,nullptr);

#ifdef TENSORLIB_CUDA_ENABLED
  thc_state = THCState_alloc();
  THCState_setDeviceAllocator(thc_state, THCCachingAllocator_get());
  thc_state->cudaHostAllocator = &THCCachingHostAllocator;
  THCudaInit(thc_state);
  generator_registry[static_cast<int>(Backend::CUDA)]
    .reset(new CUDAGenerator(this));
#endif

  generator_registry[static_cast<int>(Backend::CPU)]
    .reset(new CPUGenerator(this));
  Type::registerAll(this);
  current_default_type = &getType(Backend::CPU, ScalarType::Float);
}

Context::~Context() {
#ifdef TENSORLIB_CUDA_ENABLED
  THCState_free(thc_state);
#endif
}

Context & globalContext() {
  static Context globalContext_;
  return globalContext_;
}


}
