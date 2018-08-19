#include "ATen/Config.h"

#include "Context.h"

#include <thread>
#include <mutex>
#include <sstream>
#include <string>
#include <stdexcept>

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
: next_id(static_cast<size_t>(TypeID::NumOptions))
, thc_state(nullptr, [](THCState* p){ /* no-op */ } ) {

  THSetDefaultErrorHandler(errorHandler,nullptr);
  THSetDefaultArgErrorHandler(argErrorHandler,nullptr);

  generator_registry[static_cast<int>(DeviceType::CPU)]
    .reset(new CPUGenerator(this));
  Type::registerCPU(this);
}

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
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
