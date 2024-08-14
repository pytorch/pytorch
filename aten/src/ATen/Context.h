#pragma once

#include <ATen/BlasBackend.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/LinalgBackend.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/Generator.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HIPHooksInterface.h>
#include <ATen/detail/IPUHooksInterface.h>
#include <ATen/detail/MAIAHooksInterface.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/detail/MTIAHooksInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <c10/core/QEngine.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <cstdint>
#include <mutex>

namespace at {

class Tensor;

enum class TORCH_API Float32MatmulPrecision { HIGHEST, HIGH, MEDIUM };

class TORCH_API Context {
 public:
  Context();

  const Generator& defaultGenerator(Device device) {
    c10::DeviceType device_type = device.type();
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    if (device_type == at::kCPU) {
      return at::detail::getDefaultCPUGenerator();
    } else if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks().getDefaultCUDAGenerator(device.index());
    } else if (device_type == at::kMPS) {
      return at::detail::getMPSHooks().getDefaultMPSGenerator();
    } else if (device_type == at::kXPU) {
      return at::detail::getXPUHooks().getDefaultXPUGenerator(device.index());
    } else if (device_type == at::kIPU) {
      return at::detail::getIPUHooks().getDefaultIPUGenerator(device.index());
    } else if (device_type == at::kPrivateUse1) {
      return at::GetPrivateUse1HooksInterface()->getDefaultGenerator(
          device.index());
    } else {
      AT_ERROR(c10::DeviceTypeName(device_type), " device type not enabled.");
    }
  }
  const AcceleratorHooksInterface& getAcceleratorHooksInterface(
      std::optional<c10::DeviceType> opt_device_type = std::nullopt) {
    c10::DeviceType device_type = opt_device_type.has_value()
        ? opt_device_type.value()
        : at::getAccelerator(true).value();
    if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks();
    } else if (device_type == at::kXPU) {
      return at::detail::getXPUHooks();
    } else if (device_type == at::kMPS) {
      return at::detail::getMPSHooks();
    } else if (device_type == at::kPrivateUse1) {
      return at::detail::getPrivateUse1Hooks();
    } else if (device_type == at::kMTIA) {
      return at::detail::getMTIAHooks();
    } else if (device_type == at::kHIP) {
      return at::detail::getHIPHooks();
    } else {
      AT_ERROR(
          c10::DeviceTypeName(device_type), " device type not an accelerator.");
    }
  }
  Device getDeviceFromPtr(void* data, c10::DeviceType device_type) {
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    initXPUIfNeeded(device_type);
    if (device_type == at::kCPU) {
      return c10::DeviceType::CPU;
    } else if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks().getDeviceFromPtr(data);
    } else if (device_type == at::kXPU) {
      return at::detail::getXPUHooks().getDeviceFromPtr(data);
    } else if (device_type == at::kPrivateUse1) {
      return at::GetPrivateUse1HooksInterface()->getDeviceFromPtr(data);
    } else {
      AT_ERROR(c10::DeviceTypeName(device_type), " device type not enabled.");
    }
  }
  bool isPinnedPtr(
      const void* data,
      std::optional<c10::DeviceType> device_type = std::nullopt) {
    auto opt_device_type =
        device_type.has_value() ? device_type : at::getAccelerator();
    if (!opt_device_type.has_value() || // there is no accelerator
        !at::isAccelerator(
            opt_device_type.value())) { // passed device not an accelerator
      return false;
    }
    return getAcceleratorHooksInterface(opt_device_type.value())
        .isPinnedPtr(data);
  }
  Allocator* getPinnedMemoryAllocator(
      std::optional<c10::DeviceType> device_type = std::nullopt) {
    return getAcceleratorHooksInterface(device_type).getPinnedMemoryAllocator();
  }
  static bool hasOpenMP();
  static bool hasMKL();
  static bool hasLAPACK();
  static bool hasONEDNN();
  static bool hasMAGMA() {
    return detail::getCUDAHooks().hasMAGMA();
  }
  static bool hasCUDA() {
    return detail::getCUDAHooks().hasCUDA();
  }
  static bool hasMTIA() {
    return detail::getMTIAHooks().hasMTIA();
  }
  static bool hasCUDART() {
    return detail::getCUDAHooks().hasCUDART();
  }
  static long versionCUDART() {
    return detail::getCUDAHooks().versionCUDART();
  }
  static bool hasCuDNN() {
    return detail::getCUDAHooks().hasCuDNN();
  }
  static long versionCuDNN() {
    return detail::getCUDAHooks().versionCuDNN();
  }
  static bool hasCuSOLVER() {
    return detail::getCUDAHooks().hasCuSOLVER();
  }
  static bool hasCuBLASLt() {
    return detail::getCUDAHooks().hasCuBLASLt();
  }
  static bool hasHIP() {
    return detail::getHIPHooks().hasHIP();
  }
  static bool hasMPS() {
    return detail::getMPSHooks().hasMPS();
  }
  static bool hasIPU() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::IPU);
  }
  static bool hasXLA() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::XLA);
  }
  static bool hasXPU() {
    return detail::getXPUHooks().hasXPU();
  }
  static bool hasLazy() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::Lazy);
  }
  static bool hasMAIA() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::MAIA);
  }
  // defined in header so that getNonVariableType has ability to inline
  // call_once check. getNonVariableType is called fairly frequently
  void lazyInitCUDA() {
    c10::call_once(thc_init, [&] { detail::getCUDAHooks().initCUDA(); });
  }
  void lazyInitHIP() {
    c10::call_once(thh_init, [&] { detail::getHIPHooks().initHIP(); });
  }
  void lazyInitXPU() {
    c10::call_once(thx_init, [&] { detail::getXPUHooks().initXPU(); });
  }
  void lazyInitMTIA() {
    c10::call_once(th_mtia_init, [&] { detail::getMTIAHooks().initMTIA(); });
  }
  void lazyInitPrivateUse1() {
    c10::call_once(thp_init, [&] {
      if (isPrivateUse1HooksRegistered()) {
        at::GetPrivateUse1HooksInterface()->initPrivateUse1();
      }
    });
  }
  static const at::cuda::NVRTC& getNVRTC() {
    return detail::getCUDAHooks().nvrtc();
  }

  static bool setFlushDenormal(bool on);

  // NB: This method is *purely* whether or not a user requested
  // that CuDNN was enabled, it doesn't actually say anything about
  // whether or not CuDNN is actually usable.  Use cudnn_is_acceptable
  // to test this instead
  bool userEnabledCuDNN() const;
  void setUserEnabledCuDNN(bool e);
  bool userEnabledOnednn() const;
  void setuserEnabledOnednn(bool e);
  bool benchmarkCuDNN() const;
  void setBenchmarkCuDNN(bool);
  int benchmarkLimitCuDNN() const;
  void setBenchmarkLimitCuDNN(int);
  bool deterministicCuDNN() const;
  void setDeterministicCuDNN(bool);
  bool deterministicOnednn() const;
  void setdeterministicOnednn(bool);
  bool userEnabledNNPACK() const;
  void setUserEnabledNNPACK(bool e);

  // Note [Disabling Fused SDP Kernels]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Flash and Memory Efficient SDP kernels are enabled by default.
  // However, they can be disabled by setting
  // at::globalContext().setUserEnabledFlashSDP(false) flag.
  // This is useful for debugging purposes. For example, if you want to
  // compare the performance of the flash SDP kernels with the unfused
  // kernel, you can disable the flash SDP kernels. By disabling
  // the math SDP kernel, you can force your code to use flash kernels.
  // The math SDP kernel can be disabled by setting
  // at::globalContext().setUserEnabledMathSDP(false) flag.
  void setSDPUseFlash(bool);
  bool userEnabledFlashSDP() const;

  void setSDPUseMemEfficient(bool);
  bool userEnabledMemEfficientSDP() const;

  void setSDPUseMath(bool);
  bool userEnabledMathSDP() const;

  void setSDPUseCuDNN(bool);
  bool userEnabledCuDNNSDP() const;

  void setSDPUseOverrideable(bool);
  bool userEnabledOverrideableSDP() const;

  at::LinalgBackend linalgPreferredBackend() const;
  void setLinalgPreferredBackend(at::LinalgBackend);

  at::BlasBackend blasPreferredBackend();
  void setBlasPreferredBackend(at::BlasBackend);

  // Note [Enabling Deterministic Operations]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Operations in PyTorch that normally act nondeterministically, but have an
  // alternate deterministic implementation, should satisfy the following
  // requirements:
  //
  // * Include this comment: "See Note [Enabling Deterministic Operations]"
  //
  // * Check the value of `at::globalContext().deterministicAlgorithms()` to
  // toggle
  //   between nondeterministic and deterministic implementations.
  //
  // * Have an entry in the list of PyTorch operations that toggle between
  // nondeterministic
  //   and deterministic implementations, in the docstring of
  //   `use_deterministic_algorithms()` in torch/__init__.py
  //
  // `example_func()` below shows an example of toggling between
  // nondeterministic and deterministic implementations:
  //
  //    void example_func() {
  //      // See Note [Enabling Deterministic Operations]
  //      if (at::globalContext().deterministicAlgorithms()) {
  //        example_func_deterministic();
  //      } else {
  //        example_func_nondeterministic();
  //      }
  //    }

  bool deterministicAlgorithms() const;
  bool deterministicAlgorithmsWarnOnly() const;
  void setDeterministicAlgorithms(bool, bool);
  bool deterministicFillUninitializedMemory() const;
  void setDeterministicFillUninitializedMemory(bool);

  // Note [Writing Nondeterministic Operations]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Operations in PyTorch that act nondeterministically and do not have an
  // alternate deterministic implementation should satisfy the following
  // requirements:
  //
  // * Include this comment: "See Note [Writing Nondeterministic Operations]"
  //
  // * Include a comment explaining why the operation is nondeterministic.
  //
  // * Throw an error when `Context::deterministicAlgorithms()` is true. Most
  //   of the time, this should be accomplished by calling
  //   `at::globalContext().alertNotDeterminstic()`.  However, if the
  //   nondeterministic behavior is caused by the CuBLAS workspace
  //   configuration in CUDA >= 10.2,
  //   `at::globalContext().alertCuBLASConfigNotDeterministic()` should be
  //   called instead (in this case, a comment explaining why the operation is
  //   nondeterministic is not necessary). See below for details on these
  //   methods.
  //
  // * Have an entry in the list of nondeterministic PyTorch operations in the
  //   docstring of `use_deterministic_algorithms()` in torch/__init__.py
  //
  // * Have a test function in `test/test_torch.py` whose name begins with
  //   `test_nondeterministic_alert_`. Alternatively, if CuBLAS workspace
  //   configuration is the reason for nondeterminism, the operation should be
  //   included in the `test_cublas_config_nondeterministic_alert` test. Any new
  //   tests should ideally follow a pattern similar to the existing ones.
  //
  // `example_func()` below shows an example of the comments and error-throwing
  // code for a nondeterministic operation:
  //
  //    void example_func() {
  //      // See Note [Writing Nondeterministic Operations]
  //      // Nondeterministic because <reason>
  //      at::globalContext().alertNondeterministic("example_func");
  //      ...
  //    }

  // Throws an error if `Context::deterministicAlgorithms()` is true
  static void alertNotDeterministic(c10::string_view const& caller);

  // Throws an error if `Context::deterministicAlgorithms()` is true, CUDA
  // >= 10.2, and CUBLAS_WORKSPACE_CONFIG is not set to either ":16:8" or
  // ":4096:8". For more details:
  // https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
  void alertCuBLASConfigNotDeterministic() const;

  void setFloat32MatmulPrecision(const std::string& s);
  bool allowTF32CuDNN() const;
  void setAllowTF32CuDNN(bool);
  bool allowTF32CuBLAS() const;
  void setAllowTF32CuBLAS(bool);
  Float32MatmulPrecision float32MatmulPrecision() const;
  void setFloat32MatmulPrecision(Float32MatmulPrecision p);
  bool allowFP16ReductionCuBLAS() const;
  void setAllowFP16ReductionCuBLAS(bool);
  bool allowBF16ReductionCuBLAS() const;
  void setAllowBF16ReductionCuBLAS(bool);
  at::QEngine qEngine() const;
  void setQEngine(at::QEngine e);
  static const std::vector<at::QEngine>& supportedQEngines();
  static bool isXNNPACKAvailable();
  void setCheckSparseTensorInvariants(bool e);
  bool checkSparseTensorInvariants() const;
  // This method is used to release the original weight after pre-packing.
  // It should be called once before loading/running the model.
  // NB: By default it is set to true for mobile builds.
  void setReleaseWeightsWhenPrepacking(bool e);
  bool releaseWeightsWhenPrepacking() const;

  void setDisplayVmapFallbackWarnings(bool enabled);
  bool areVmapFallbackWarningsEnabled() const;

  void setDefaultMobileCPUAllocator();
  void unsetDefaultMobileCPUAllocator();
  bool allowFP16ReductionCPU() const;
  void setAllowFP16ReductionCPU(bool);

 private:
  void initCUDAIfNeeded(c10::DeviceType p) {
    if (p == c10::DeviceType::CUDA) {
      lazyInitCUDA();
    }
  }
  void initHIPIfNeeded(c10::DeviceType p) {
    if (p == c10::DeviceType::HIP) {
      lazyInitHIP();
    }
  }
  void initXPUIfNeeded(c10::DeviceType p) {
    if (p == c10::DeviceType::XPU) {
      lazyInitXPU();
    }
  }
  static bool checkCuBLASConfigDeterministic();
  c10::once_flag thc_init;
  c10::once_flag thh_init;
  c10::once_flag thx_init;
  c10::once_flag th_mtia_init;
  c10::once_flag thp_init;
  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool deterministic_onednn = false;
  bool _deterministic_algorithms = false;
  bool _deterministic_algorithms_warn_only = false;
  bool _deterministic_fill_uninitialized_memory = true;
  bool enabled_flashSDP = true;
  bool enabled_mem_efficientSDP = true;
  bool enabled_mathSDP = true;
  bool enabled_cudnnSDP = true;
  bool enabled_overrideable = true;
#ifdef USE_ROCM
  bool benchmark_cudnn = true;
#else
  bool benchmark_cudnn = false;
#endif
  Float32MatmulPrecision float32_matmul_precision =
      c10::utils::check_env("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == true
      ? at::Float32MatmulPrecision::HIGH
      : at::Float32MatmulPrecision::HIGHEST;
  int benchmark_limit_cudnn = 10;
  bool allow_tf32_cudnn = true;
  bool allow_fp16_reduction_cublas = true;
  bool allow_bf16_reduction_cublas = true;
  bool enabled_onednn = true;
  bool enabled_nnpack = true;
  at::LinalgBackend linalg_preferred_backend =
      c10::utils::check_env("TORCH_LINALG_PREFER_CUSOLVER") == true
      ? at::LinalgBackend::Cusolver
      : at::LinalgBackend::Default;
  at::BlasBackend blas_preferred_backend =
#ifdef USE_ROCM
      (c10::utils::check_env("TORCH_BLAS_PREFER_HIPBLASLT") != false)
#else
      (c10::utils::check_env("TORCH_BLAS_PREFER_CUBLASLT") == true)
#endif
      ? at::BlasBackend::Cublaslt
      : at::BlasBackend::Cublas;
#ifdef C10_MOBILE
  bool release_original_weights = true;
#else
  bool release_original_weights = false;
#endif
  bool display_vmap_fallback_warnings_ = false;
  std::optional<at::QEngine> quantized_engine = std::nullopt;
  bool enable_sparse_tensor_invariant_checks = false;
  bool allow_fp16_reduction_cpu = false;

  Allocator* prev_allocator_ptr_{nullptr};
};

TORCH_API Context& globalContext();

inline void init() {
  globalContext();
}

TORCH_API Allocator* getCPUAllocator();

inline DeprecatedTypeProperties& getDeprecatedTypeProperties(
    Backend p,
    ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      p, s);
}

inline DeprecatedTypeProperties& CPU(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::CPU, s);
}

inline DeprecatedTypeProperties& CUDA(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::CUDA, s);
}

inline DeprecatedTypeProperties& HIP(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::HIP, s);
}

inline DeprecatedTypeProperties& MPS(ScalarType s) {
  return globalDeprecatedTypePropertiesRegistry().getDeprecatedTypeProperties(
      Backend::MPS, s);
}

inline bool hasCUDA() {
  return globalContext().hasCUDA();
}

inline bool hasMTIA() {
  return globalContext().hasMTIA();
}

inline bool hasHIP() {
  return globalContext().hasHIP();
}

inline bool hasIPU() {
  return globalContext().hasIPU();
}

inline bool hasXLA() {
  return globalContext().hasXLA();
}

inline bool hasMPS() {
  return globalContext().hasMPS();
}

inline bool hasMAIA() {
  return globalContext().hasMAIA();
}

inline bool hasXPU() {
  return globalContext().hasXPU();
}

// Despite its name, this function returns the number of *CUDA* GPUs.
inline size_t getNumGPUs() {
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

inline bool hasOpenMP() {
  return globalContext().hasOpenMP();
}

inline bool hasMKL() {
  return globalContext().hasMKL();
}

inline bool hasLAPACK() {
  return globalContext().hasLAPACK();
}

inline bool hasMAGMA() {
  return globalContext().hasMAGMA();
}

inline bool hasONEDNN() {
  return globalContext().hasONEDNN();
}

inline void manual_seed(uint64_t seed) {
  auto gen = globalContext().defaultGenerator(c10::DeviceType::CPU);
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_current_seed(seed);
  }
  // NB: Sometimes we build with CUDA, but we don't have any GPUs
  // available. In that case, we must not seed CUDA; it will fail!
  const auto cuda_num_gpus = detail::getCUDAHooks().getNumGPUs();
  if (hasCUDA() && cuda_num_gpus > 0) {
    for (const auto i : c10::irange(cuda_num_gpus)) {
      auto cuda_gen = globalContext().defaultGenerator(
          Device(at::kCUDA, static_cast<c10::DeviceIndex>(i)));
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(cuda_gen.mutex());
        cuda_gen.set_current_seed(seed);
      }
    }
  }

  const auto xpu_num_gpus = detail::getXPUHooks().getNumGPUs();
  if (hasXPU() && xpu_num_gpus) {
    for (const auto i : c10::irange(xpu_num_gpus)) {
      auto xpu_gen = globalContext().defaultGenerator(
          Device(at::kXPU, static_cast<c10::DeviceIndex>(i)));
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(xpu_gen.mutex());
        xpu_gen.set_current_seed(seed);
      }
    }
  }

  if (hasMPS()) {
    auto mps_gen = globalContext().defaultGenerator(c10::DeviceType::MPS);
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(mps_gen.mutex());
    mps_gen.set_current_seed(seed);
  }
}

// When the global flag `allow_tf32` is set to true, cuBLAS handles are
// automatically configured to use math mode CUBLAS_TF32_TENSOR_OP_MATH.
// For some operators, such as addmv, TF32 offers no performance improvement
// but causes precision loss. To help this case, this class implements
// a RAII guard that can be used to quickly disable TF32 within its scope.
//
// Usage:
//     NoTF32Guard disable_tf32;
struct TORCH_API NoTF32Guard {
  NoTF32Guard();
  ~NoTF32Guard();
  static bool should_disable_tf32();

 private:
  bool changed = false;
};

struct TORCH_API ROCmBackwardPassGuard {
  ROCmBackwardPassGuard();
  ~ROCmBackwardPassGuard();
  static bool is_backward_pass();
};

} // namespace at
