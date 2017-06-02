#include "Context.h"
#include <thread>
#include <mutex>

namespace tlib {

Context::Context() {
  thc_state = THCState_alloc();
  THCState_setDeviceAllocator(thc_state, THCCachingAllocator_get());
  thc_state->cudaHostAllocator = &THCCachingHostAllocator;
  THCudaInit(thc_state);
  cuda_gen.reset(new CUDAGenerator(this));
  cpu_gen.reset(new CPUGenerator(this));
}

Context::~Context() {
  THCState_free(thc_state);
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
