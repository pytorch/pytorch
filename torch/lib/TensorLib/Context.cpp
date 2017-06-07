#include "Context.h"
#include <thread>
#include <mutex>

#ifdef TENSORLIB_HAS_CUDA
#include "THC/THC.h"
#include "TensorLib/CUDAGenerator.h"
#endif
#include "TensorLib/CPUGenerator.h"

namespace tlib {

Context::Context() {

#ifdef TENSORLIB_HAS_CUDA
  thc_state = THCState_alloc();
  THCState_setDeviceAllocator(thc_state, THCCachingAllocator_get());
  thc_state->cudaHostAllocator = &THCCachingHostAllocator;
  THCudaInit(thc_state);
  generator_registry[static_cast<int>(Processor::CUDA)]
    .reset(new CUDAGenerator(this));
#endif

  generator_registry[static_cast<int>(Processor::CPU)]
    .reset(new CPUGenerator(this));
  Type::registerAll(this);
}

Context::~Context() {
#ifdef TENSORLIB_HAS_CUDA
  THCState_free(thc_state);
#endif
}

static std::mutex context_lock;
static Context * globalContext_ = nullptr;

Context * globalContext() {
  if(!globalContext_) {
    std::lock_guard<std::mutex> lock(context_lock);
    if (!globalContext_) {
      globalContext_ = new Context();
    }
  }
  return globalContext_;
}


}
