#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <ATen/CPUGenerator.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HIPHooksInterface.h>
#include <c10/util/Exception.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/QEngine.h>

#include <memory>
#include <mutex>
#include <cstdint>

namespace at {

class Tensor;

class CAFFE2_API Context {
 public:
  Context();

  Generator & defaultGenerator(Device device) {
    DeviceType device_type = device.type();
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    if (device_type == at::kCPU) {
      return *at::detail::getDefaultCPUGenerator();
    } else if (device_type == at::kCUDA) {
      return *at::detail::getCUDAHooks().getDefaultCUDAGenerator(device.index());
    } else {
      AT_ERROR(DeviceTypeName(device_type), " device type not enabled.");
    }
  }
  Device getDeviceFromPtr(void* data, DeviceType device_type) {
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    if (device_type == at::kCPU) {
      return DeviceType::CPU;
    } else if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks().getDeviceFromPtr(data);
    } else {
      AT_ERROR(DeviceTypeName(device_type), " device type not enabled.");
    }
  }
  bool isPinnedPtr(void* data) {
    return detail::getCUDAHooks().isPinnedPtr(data);
  }
  bool hasOpenMP() const;
  bool hasMKL() const;
  bool hasLAPACK() const;
  bool hasMKLDNN() const;
  bool hasMAGMA() const {
    return detail::getCUDAHooks().hasMAGMA();
  }
  bool hasCUDA() const {
    return detail::getCUDAHooks().hasCUDA();
  }
  bool hasHIP() const {
    return detail::getHIPHooks().hasHIP();
  }
  bool hasXLA() const {
    return c10::impl::hasDeviceGuardImpl(at::DeviceType::XLA);
  }
  // defined in header so that getNonVariableType has ability to inline
  // call_once check. getNonVariableType is called fairly frequently
  THCState* lazyInitCUDA() {
    std::call_once(thc_init,[&] {
      thc_state = detail::getCUDAHooks().initCUDA();
    });
    return thc_state.get();
  }
  THHState* lazyInitHIP() {
    std::call_once(thh_init,[&] {
      thh_state = detail::getHIPHooks().initHIP();
    });
    return thh_state.get();
  }
  const at::cuda::NVRTC& getNVRTC() {
    return detail::getCUDAHooks().nvrtc();
  }
  THCState* getTHCState() {
    // AT_ASSERT(thc_state);
    return thc_state.get();
  }
  THHState* getTHHState() {
    return thh_state.get();
  }

  bool setFlushDenormal(bool on);

  // NB: This method is *purely* whether or not a user requested
  // that CuDNN was enabled, it doesn't actually say anything about
  // whether or not CuDNN is actually usable.  Use cudnn_is_acceptable
  // to test this instead
  bool userEnabledCuDNN() const;
  void setUserEnabledCuDNN(bool e);
  bool userEnabledMkldnn() const;
  void setUserEnabledMkldnn(bool e);
  bool benchmarkCuDNN() const;
  void setBenchmarkCuDNN(bool);
  bool deterministicCuDNN() const;
  void setDeterministicCuDNN(bool);
  at::QEngine qEngine() const;
  void setQEngine(at::QEngine e);
  const std::vector<at::QEngine>& supportedQEngines() const;

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
  std::once_flag thc_init;
  std::once_flag thh_init;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool benchmark_cudnn = false;
  bool enabled_mkldnn = true;
  c10::optional<at::QEngine> quantized_engine = c10::nullopt;
  std::unique_ptr<THCState, void(*)(THCState*)> thc_state;
  std::unique_ptr<THHState, void(*)(THHState*)> thh_state;
};

CAFFE2_API Context& globalContext();

static inline void init() {
  globalContext();
}

CAFFE2_API Allocator* getCPUAllocator();

static inline DeprecatedTypeProperties& getNonVariableDeprecatedTypeProperties(Backend p, ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      p, s, /*is_variable*/false);
}

static inline DeprecatedTypeProperties& CPU(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::CPU, s, /*is_variable*/false);
}

static inline DeprecatedTypeProperties& CUDA(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::CUDA, s, /*is_variable*/false);
}

static inline DeprecatedTypeProperties& HIP(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::HIP, s, /*is_variable*/false);
}

static inline bool hasCUDA() {
  return globalContext().hasCUDA();
}

static inline bool hasHIP() {
  return globalContext().hasHIP();
}

static inline bool hasXLA() {
  return globalContext().hasXLA();
}

// Despite its name, this function returns the number of *CUDA* GPUs.
static inline size_t getNumGPUs() {
  // WARNING: DO NOT ADD LOGIC TO HANDLE OTHER DEVICE TYPES TO THIS
  // FUNCTION.  If you are interested in interrogating the number of
  // devices for a specific device type, add that function to the
  // relevant library (e.g., similar to at::cuda::device_count())
  if (hasCUDA() && hasHIP()) {
    throw std::runtime_error(
        "Enabling both CUDA and HIP in ATen is not supported, as HIP masquerades "
        "to be CUDA (e.g., when you say CUDA, on a HIP build of ATen, this actually "
        "means HIP.  Rebuild PyTorch with one or the other disabled.");
  } else if (hasCUDA()) {
    return detail::getCUDAHooks().getNumGPUs();
  } else if (hasHIP()) {
    return detail::getHIPHooks().getNumGPUs();
  } else {
    return 0;
  }
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

static inline bool hasMKLDNN() {
  return globalContext().hasMKLDNN();
}

static inline void manual_seed(uint64_t seed) {
  auto& gen = globalContext().defaultGenerator(DeviceType::CPU);
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex_);
    gen.set_current_seed(seed);
  }
  // NB: Sometimes we build with CUDA, but we don't have any GPUs
  // available. In that case, we must not seed CUDA; it will fail!
  int num_gpus = detail::getCUDAHooks().getNumGPUs();
  if (hasCUDA() && num_gpus > 0) {
    for (int i = 0; i < num_gpus; i++) {
      auto& cuda_gen = globalContext().defaultGenerator(Device(at::kCUDA, i));
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(cuda_gen.mutex_);
        cuda_gen.set_current_seed(seed);
      }
    }
  }
}

} // namespace at
