#pragma once

#include "ATen/ATenGeneral.h"
#include <ATen/CPUGeneral.h>
#include "ATen/Generator.h"
#include "ATen/Type.h"
#include "ATen/Utils.h"
#include "ATen/Error.h"
#include "ATen/detail/CUDAHooksInterface.h"
#include "ATen/CUDAStream.h"

#include <memory>
#include <mutex>
#include <cstdint>

namespace at {

enum class IsVariable {
  NotVariable,
  Variable,
  NumOptions
};

class AT_API Context {
public:
  Context();
  Type* getTypeRaw(Backend p, ScalarType s) {
    return type_registry[static_cast<int>(p)][static_cast<int>(s)].get();
  }
  Type * getTypeOpt(Backend p, ScalarType s) {
    initCUDAIfNeeded(p);
    auto type = getTypeRaw(p, s);

    if(!type) {
      // there is only a single Undefined Type.
      if (p == Backend::Undefined || s == ScalarType::Undefined) {
        return getTypeRaw(Backend::Undefined, ScalarType::Undefined);
      }
    }

    return type;
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

  CUDAStream createCUDAStream() const {
    return detail::CUDAStream_createAndRetainWithOptions(
      CUDAStream::DEFAULT_FLAGS
    , CUDAStream::DEFAULT_PRIORITY
    );
  }

  CUDAStream createCUDAStreamWithOptions(int32_t flags, int32_t priority) const {
    return detail::CUDAStream_createAndRetainWithOptions(flags, priority);
  }

  CUDAStream getDefaultCUDAStream() const {
    return detail::CUDAStream_getDefaultStream();
  }

  CUDAStream getDefaultCUDAStreamOnDevice(int64_t device) const {
    return detail::CUDAStream_getDefaultStreamOnDevice(device);
  }

  CUDAStream getCurrentCUDAStream() const {
    return detail::CUDAStream_getAndRetainCurrentStream();
  }

  CUDAStream getCurrentCUDAStreamOnDevice(int64_t device) const {
    return detail::CUDAStream_getAndRetainCurrentStreamOnDevice(device);
  }

  void setCurrentCUDAStream(CUDAStream stream) const {
    return detail::CUDAStream_setStream(stream.internals());
  }

  void setCurrentCUDAStreamOnDevice(int64_t device, CUDAStream stream) const {
    return detail::CUDAStream_setStreamOnDevice(device, stream.internals());
  }

#ifndef __HIP_PLATFORM_HCC__
  cusparseHandle_t getCurrentCUDASparseHandle() const {
    return detail::getCUDAHooks().getCurrentCUDASparseHandle(thc_state.get());
  }
#endif
  cudaDeviceProp* getCurrentDeviceProperties() const {
    return detail::getCUDAHooks().getCurrentDeviceProperties(thc_state.get());
  }
  cudaDeviceProp* getDeviceProperties(int device) const {
    return detail::getCUDAHooks().getDeviceProperties(thc_state.get(), device);
  }
  int getNumGPUs() const {
    return detail::getCUDAHooks().getNumGPUs();
  }
  size_t freshTypeID() {
    return next_id++;
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
private:
  // NB: type_registry has nullptr for all CUDA backends until
  // CUDA initialization has occurred
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  void initCUDAIfNeeded(Backend p) {
    if(p == Backend::CUDA)
      lazyInitCUDA();
  }
  std::once_flag thc_init;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool benchmark_cudnn = false;
  std::atomic<size_t> next_id;
  std::unique_ptr<THCState, void(*)(THCState*)> thc_state;
  friend struct Type;
  friend void register_cuda_types(Context * context);
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
