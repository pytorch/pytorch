#pragma once

#include "ATen/ATenGeneral.h"
#include <ATen/CPUGeneral.h>
#include "ATen/Generator.h"
#include "ATen/Type.h"
#include "ATen/Utils.h"
#include "ATen/Error.h"
#include "ATen/detail/CUDAHooksInterface.h"

#include <memory>
#include <mutex>
#include <cstdint>

// Forwarde declare these CUDA types here to avoid including CUDA headers in
// ATen headers, which would make ATen always require CUDA to build.
struct THCState;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;
struct cudaDeviceProp;

namespace at {

enum class IsVariable {
  NotVariable,
  Variable,
  NumOptions
};

class AT_API Context {
public:
  Context();
  Type * getTypeOpt(Backend p, ScalarType s) {
    initCUDAIfNeeded(p);
    auto & type = type_registry[static_cast<int>(IsVariable::NotVariable)][static_cast<int>(p)][static_cast<int>(s)];

    if(!type) {
      // there is only a single Undefined Type.
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        auto & undef = type_registry[static_cast<int>(IsVariable::NotVariable)][static_cast<int>(Backend::Undefined)][static_cast<int>(ScalarType::Undefined)];
        if (undef) return undef.get();
      }
      return nullptr;
    }
    return type.get();
  }
  Type & getType(Backend p, ScalarType s) {
    auto* type = getTypeOpt(p, s);
    if (!type) AT_ERROR(toString(p), toString(s), "Type is not enabled.");
    return *type;
  }
  Generator & defaultGenerator(Backend p) {
    initCUDAIfNeeded(p);
    auto & generator = generator_registry[static_cast<int>(p)];
    if(!generator)
      AT_ERROR(toString(p), " backend type not enabled.");
    return *generator;
  }
  bool hasMKL() const;
  bool hasCUDA() const {
    return detail::getCUDAHooks().hasCUDA();
  }
  bool hasCuDNN() const {
    return detail::getCUDAHooks().hasCuDNN();
  }
  int64_t current_device() const {
    return detail::getCUDAHooks().current_device();
  }
  // defined in header so that getType has ability to inline
  // call_once check. getType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      thc_state = detail::getCUDAHooks().initCUDA();
      generator_registry[static_cast<int>(Backend::CUDA)] =
        detail::getCUDAHooks().initCUDAGenerator(this);
      detail::getCUDAHooks().registerCUDATypes(this);
    });
    return thc_state.get();
  }

  THCState* getTHCState() {
    // AT_ASSERT(thc_state);
    return thc_state.get();
  }

  cudaStream_t getCurrentCUDAStream() const {
    return detail::getCUDAHooks().getCurrentCUDAStream(thc_state.get());
  }
  cudaDeviceProp* getCurrentDeviceProperties() const {
    return detail::getCUDAHooks().getCurrentDeviceProperties(thc_state.get());
  }
  cudaDeviceProp* getDeviceProperties(int device) const {
    return detail::getCUDAHooks().getDeviceProperties(thc_state.get(), device);
  }
  int getNumGPUs() const {
    return detail::getCUDAHooks().getNumGPUs();
  }

  bool setFlushDenormal(bool on);

  // NB: This method is *purely* whether or not a user requested
  // that CuDNN was enabled, it doesn't actually say anything about
  // whether or not CuDNN is actually usable.  Use cudnn_is_acceptable
  // to test this instead
  bool userEnabledCuDNN() const;
  void setUserEnabledCuDNN(bool e);
  bool benchmarkCuDNN() const;
  void setBenchmarkCuDNN(bool);
  bool deterministicCuDNN() const;
  void setDeterministicCuDNN(bool);
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(Backend::NumOptions)];
  // NB: type_registry has nullptr for all CUDA backends until
  // CUDA initialization has occurred
  std::unique_ptr<Type> type_registry
    [static_cast<int>(IsVariable::NumOptions)]
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
private:
  void initCUDAIfNeeded(Backend p) {
    if(p == Backend::CUDA)
      lazyInitCUDA();
  }
  std::once_flag thc_init;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool benchmark_cudnn = false;
  std::unique_ptr<THCState, void(*)(THCState*)> thc_state;
};

AT_API Context & globalContext();

static inline void init() {
  globalContext();
  if (const char *env_p = std::getenv("OMP_NUM_THREADS")) {
    at::set_num_threads(std::stoi(env_p));
  }
  if (const char *env_p = std::getenv("MKL_NUM_THREADS")) {
    at::set_num_threads(std::stoi(env_p));
  }
}

static inline Type& getType(Backend p, ScalarType s) {
  return globalContext().getType(p, s);
}

static inline Type& CPU(ScalarType s) {
  return getType(Backend::CPU, s);
}

static inline Type& CUDA(ScalarType s) {
  return getType(Backend::CUDA, s);
}

static inline bool hasCUDA() {
  return globalContext().hasCUDA();
}

static inline bool hasCuDNN() {
  return globalContext().hasCuDNN();
}

static inline bool hasMKL() {
  return globalContext().hasMKL();
}

static inline int64_t current_device() {
  return globalContext().current_device();
}

} // namespace at
