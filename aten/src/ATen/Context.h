#pragma once

#include <ATen/CPUGeneral.h>
#include <ATen/Type.h>
#include <ATen/TypeExtendedInterface.h>
#include <ATen/Utils.h>
#include <ATen/LegacyTHDispatch.h>
#include <ATen/LegacyTHDispatcher.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HIPHooksInterface.h>
#include <ATen/detail/ComplexHooksInterface.h>
#include <c10/util/Exception.h>

#include <memory>
#include <mutex>
#include <cstdint>

namespace at {

class Tensor;

class CAFFE2_API Context {
 public:
  Context();
  TypeExtendedInterface* getNonVariableTypeRaw(Backend p, ScalarType s) {
    return static_cast<TypeExtendedInterface*>(globalLegacyTypeDispatch().getNonVariableTypeRaw(p, s));
  }
  TypeExtendedInterface * getNonVariableTypeOpt(Backend p, ScalarType s) {
    return static_cast<TypeExtendedInterface*>(globalLegacyTypeDispatch().getNonVariableTypeOpt(p, s));
  }
  TypeExtendedInterface & getNonVariableType(Backend p, ScalarType s) {
    return static_cast<TypeExtendedInterface&>(globalLegacyTypeDispatch().getNonVariableType(p, s));
  }
  TypeExtendedInterface & getVariableType(Backend p, ScalarType s) {
    return static_cast<TypeExtendedInterface&>(globalLegacyTypeDispatch().getVariableType(p, s));
  }
  TypeExtendedInterface & getType(Backend p, ScalarType s, bool is_variable) {
    return static_cast<TypeExtendedInterface&>(globalLegacyTypeDispatch().getType(p, s, is_variable));
  }
  LegacyTHDispatcher& getLegacyTHDispatcher(Backend p, ScalarType s) {
    return globalLegacyTHDispatch().getLegacyTHDispatcher(p, s);
  }
  // The passed in Type must be delete'able
  // TODO: Just make it take a unique_ptr
  void registerType(Backend b, ScalarType s, Type* t) {
    globalLegacyTypeDispatch().registerType(b, s,
      LegacyTypeDispatch::TypeUniquePtr{t, LegacyTypeDeleter([](Type* p) { delete p; }) });
  }

  void registerLegacyTHDispatcher(Backend b, ScalarType s, LegacyTHDispatcher* t) {
    globalLegacyTHDispatch().registerDispatcher(b, s,
      LegacyTHDispatch::LegacyTHDispatcherUniquePtr{t, LegacyTHDispatcherDeleter([](LegacyTHDispatcher* p) { delete p; }) });
  }

  Generator & defaultGenerator(DeviceType device_type) {
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    auto & generator = generator_registry[static_cast<int>(device_type)];
    if(!generator)
      AT_ERROR(DeviceTypeName(device_type), " backend type not enabled.");
    return *generator;
  }
  bool hasOpenMP() const;
  bool hasMKL() const;
  bool hasLAPACK() const;
  bool hasMAGMA() const {
    return detail::getCUDAHooks().hasMAGMA();
  }
  bool hasCUDA() const {
    return detail::getCUDAHooks().hasCUDA();
  }
  bool hasHIP() const {
    return detail::getHIPHooks().hasHIP();
  }
  // defined in header so that getNonVariableType has ability to inline
  // call_once check. getNonVariableType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      thc_state = detail::getCUDAHooks().initCUDA();
      generator_registry[static_cast<int>(DeviceType::CUDA)] =
        detail::getCUDAHooks().initCUDAGenerator(this);
      detail::getCUDAHooks().registerCUDATypes(this);
    });
    return thc_state.get();
  }
  THHState* lazyInitHIP() {
    std::call_once(thh_init,[&] {
      thh_state = detail::getHIPHooks().initHIP();
      generator_registry[static_cast<int>(DeviceType::HIP)] =
        detail::getHIPHooks().initHIPGenerator(this);
      detail::getHIPHooks().registerHIPTypes(this);
    });
    return thh_state.get();
  }
  void lazyInitComplex() {
    std::call_once(complex_init_, [&] {
      detail::getComplexHooks().registerComplexTypes(this);
    });
  }

  THCState* getTHCState() {
    // AT_ASSERT(thc_state);
    return thc_state.get();
  }
  THHState* getTHHState() {
    return thh_state.get();
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
    generator_registry[static_cast<int>(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];
private:
  void initCUDAIfNeeded(DeviceType p) {
    if (p == DeviceType::CUDA) {
      lazyInitCUDA();
    }
  }
  void initHIPIfNeeded(DeviceType p) {
    if (p == DeviceType::HIP) {
      lazyInitHIP();
    }
  }
  void initComplexIfNeeded(ScalarType s) {
    if (isComplexType(s)) {
      lazyInitComplex();
    }
  }
  std::once_flag thc_init;
  std::once_flag thh_init;
  std::once_flag complex_init_;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool benchmark_cudnn = false;
  std::atomic<size_t> next_id;
  std::unique_ptr<THCState, void(*)(THCState*)> thc_state;
  std::unique_ptr<THHState, void(*)(THHState*)> thh_state;
  friend struct Type;
};

CAFFE2_API Context& globalContext();

static inline void init() {
  globalContext();
  if (const char *env_p = std::getenv("OMP_NUM_THREADS")) {
    at::set_num_threads(std::stoi(env_p));
  }
  if (const char *env_p = std::getenv("MKL_NUM_THREADS")) {
    at::set_num_threads(std::stoi(env_p));
  }
}

static inline TypeExtendedInterface& getNonVariableType(Backend p, ScalarType s) {
  return globalContext().getNonVariableType(p, s);
}

CAFFE2_API TypeExtendedInterface& getType(TensorOptions options);
CAFFE2_API TypeExtendedInterface& getType(const TensorImpl*);
CAFFE2_API TypeExtendedInterface& getType(const Tensor&);

CAFFE2_API Allocator* getCPUAllocator();

static inline TypeExtendedInterface& CPU(ScalarType s) {
  return getNonVariableType(Backend::CPU, s);
}

static inline TypeExtendedInterface& CUDA(ScalarType s) {
  return getNonVariableType(Backend::CUDA, s);
}

static inline TypeExtendedInterface& HIP(ScalarType s) {
  return getNonVariableType(Backend::HIP, s);
}

CAFFE2_API LegacyTHDispatcher& getLegacyTHDispatcher(TensorOptions options);
CAFFE2_API LegacyTHDispatcher& getLegacyTHDispatcher(const Tensor&);

static inline bool hasCUDA() {
  return globalContext().hasCUDA();
}

static inline bool hasHIP() {
  return globalContext().hasHIP();
}

static inline size_t getNumGPUs() {
  if (hasCUDA()) {
    return detail::getCUDAHooks().getNumGPUs();
  }
  if (hasHIP()) {
    return detail::getHIPHooks().getNumGPUs();
  }
  return 0;
}

static inline bool hasOpenMP() {
  return globalContext().hasOpenMP();
}

static inline bool hasMKL() {
  return globalContext().hasMKL();
}

static inline bool hasLAPACK() {
  return globalContext().hasLAPACK();
}

static inline bool hasMAGMA() {
  return globalContext().hasMAGMA();
}

static inline void manual_seed(uint64_t seed) {
  globalContext().defaultGenerator(DeviceType::CPU).manualSeed(seed);
  // NB: Sometimes we build with CUDA, but we don't have any GPUs
  // available. In that case, we must not seed CUDA; it will fail!
  if (hasCUDA() && detail::getCUDAHooks().getNumGPUs() > 0) {
    globalContext().defaultGenerator(DeviceType::CUDA).manualSeedAll(seed);
  }
}

} // namespace at
