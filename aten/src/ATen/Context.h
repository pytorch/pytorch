#pragma once

#include "ATen/ATenGeneral.h"
#include <ATen/CPUGeneral.h>
#include "ATen/Generator.h"
#include "ATen/Type.h"
#include "ATen/Utils.h"
#include "ATen/Error.h"

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

class AT_API Context {
public:
  Context();
  Type & getType(Backend p, ScalarType s) {
    initCUDAIfNeeded(p);
    auto & type = type_registry[static_cast<int>(p)][static_cast<int>(s)];

    if(!type) {
      // there is only a single Undefined Type.
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        auto & undef = type_registry[static_cast<int>(Backend::Undefined)][static_cast<int>(ScalarType::Undefined)];
        if (undef) return *undef;
      }
      AT_ERROR("%s%sType is not enabled.",toString(p),toString(s));
    }
    return *type;
  }
  Generator & defaultGenerator(Backend p) {
    initCUDAIfNeeded(p);
    auto & generator = generator_registry[static_cast<int>(p)];
    if(!generator)
      AT_ERROR("%s backend type not enabled.",toString(p));
    return *generator;
  }
  bool hasMKL() const;
  bool hasCUDA() const;
  int64_t current_device() const;
  // defined in header so that getType has ability to inline
  // call_once check. getType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      doInitCUDA();
    });
    return thc_state;
  }

  cudaStream_t getCurrentCUDAStream() const;
  cudaDeviceProp* getCurrentDeviceProperties() const;
  cudaDeviceProp* getDeviceProperties(int device) const;

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
  ~Context();
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(Backend::NumOptions)];
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  // TODO: Consider making this private
  THCState * thc_state;
private:
  void initCUDAIfNeeded(Backend p) {
    if(p == Backend::CUDA)
      lazyInitCUDA();
  }
  void doInitCUDA();
  std::once_flag thc_init;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool benchmark_cudnn = false;
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

static inline bool hasMKL() {
  return globalContext().hasMKL();
}

static inline int64_t current_device() {
  return globalContext().current_device();
}

} // namespace at
