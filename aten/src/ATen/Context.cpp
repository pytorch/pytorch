#include "ATen/Config.h"

#include "Context.h"

#include <ATen/core/TensorOptions.h>

#include <thread>
#include <mutex>
#include <sstream>
#include <string>
#include <stdexcept>

#include "ATen/CPUGenerator.h"
#include "ATen/RegisterCPU.h"
#include "ATen/Tensor.h"
#include <ATen/cpu/FlushDenormal.h>

#include "TH/TH.h"  // for USE_LAPACK

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
  register_cpu_types(this);
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

bool Context::hasLAPACK() const {
#ifdef USE_LAPACK
  return true;
#else
  return false;
#endif
}

bool Context::setFlushDenormal(bool on) {
  return at::cpu::set_flush_denormal(on);
}

TypeExtendedInterface& getType(TensorOptions options) {
  return globalContext().getType(
            options.backend(), typeMetaToScalarType(options.dtype()), options.is_variable());
}

TypeExtendedInterface& getType(const TensorImpl* impl) {
  Backend backend = tensorTypeIdToBackend(impl->type_id());
  return globalContext().getType(
            backend, typeMetaToScalarType(impl->dtype()), impl->is_variable());
}

TypeExtendedInterface& getType(const Tensor& t) {
  return getType(t.unsafeGetTensorImpl());
}

Allocator* getCPUAllocator() {
  return getTHDefaultAllocator();
}

struct LegacyTypeInit : public LegacyTypeInitInterface {
  LegacyTypeInit(LegacyTypeInitArgs) {}
  void initCPU() const override {
    globalContext();
  }
  void initCUDA() const override {
    globalContext().lazyInitCUDA();
  }
  void initComplex() const override {
    globalContext().lazyInitComplex();
  }
};
REGISTER_LEGACY_TYPE_INIT(LegacyTypeInit);

}
