#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif
#include <cstdlib>
#include <cstring>

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
#include <sys/auxv.h>
#endif

namespace at::native {

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
static inline bool cpu_has_vxe()
{
  return (getauxval(AT_HWCAP) & HWCAP_S390_VXE);
}
#endif

static CPUCapability compute_cpu_capability() {
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
#if defined(HAVE_VSX_CPU_DEFINITION)
    if (strcmp(envar, "vsx") == 0) {
      return CPUCapability::VSX;
    }
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
    if (strcmp(envar, "zvector") == 0) {
      return CPUCapability::ZVECTOR;
    }
#elif defined(HAVE_SVE_CPU_DEFINITION)
    int sve_vl = cpuinfo_get_max_arm_sve_length(); //Returns maximum SVE VL supported by your HW.
    if (strcmp(envar, "sve") == 0) {
      if (sve_vl == 512) {
        return CPUCapability::SVE512;
      } else if (sve_vl == 256) {
        return CPUCapability::SVE256;
      } else if (sve_vl == 128) {
        return CPUCapability::SVE128;
      } else {
        TORCH_WARN("SVE capability not available on hardware. Falling back to DEFAULT");
        return CPUCapability::DEFAULT;
      }
    }
#else
#ifdef HAVE_AVX512_CPU_DEFINITION
    if (strcmp(envar, "avx512") == 0) {
      return CPUCapability::AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (strcmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }
#endif
#endif
    if (strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
  }

#if !defined(__powerpc__) && !defined(__s390x__) && !defined(HAVE_SVE_CPU_DEFINITION)
  if (cpuinfo_initialize()) {
#if defined(HAVE_AVX512_CPU_DEFINITION)
    // GCC supports some AVX512 intrinsics such as _mm512_set_epi16 only in
    // versions 9 & beyond. So, we want to ensure that only releases built with
    // supported compilers on supported hardware return CPU Capability AVX512,
    // if it's supported on the hardware PyTorch is running on.
    if (cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() &&  \
        cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX2;
    }
#endif
  }
#endif

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  // vxe is needed for fp32 vector instructions
  if (cpu_has_vxe()) {
    return CPUCapability::ZVECTOR;
  }
#endif

#if defined(__linux__) && defined(HAVE_SVE_CPU_DEFINITION)
  if (cpuinfo_initialize() && cpuinfo_has_arm_sve()) {
    int sve_vl = cpuinfo_get_max_arm_sve_length(); //Returns maximum SVE VL supported by your HW.
    if (sve_vl <= 0) {
      // SVE is not supported on this system.
      // Return the default CPU capability.
      return CPUCapability::DEFAULT;
    }
    #ifdef HAVE_SVE512_CPU_DEFINITION
        if (sve_vl == 512) { // Check for SVE512
            return CPUCapability::SVE512;
        }
    #endif
    #ifdef HAVE_SVE256_CPU_DEFINITION
        if (sve_vl == 256) { // Check for SVE256
            return CPUCapability::SVE256;
        }
    #endif
    #ifdef HAVE_SVE128_CPU_DEFINITION
        if (sve_vl == 128) { // Check for SVE128
            return CPUCapability::SVE128;
        }
    #endif
    // Return the default CPU capability.
    return CPUCapability::DEFAULT;
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  return CPUCapability::VSX;
#else
  return CPUCapability::DEFAULT;
#endif
}

CPUCapability get_cpu_capability() {
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

DispatchResult DispatchStubImpl::try_get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
  , void *SVE512
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  , void *SVE256
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
  , void *SVE128
#endif
) {
  constexpr auto supported_devices = c10::array_of<c10::DeviceType>(
        c10::DeviceType::CPU,
        c10::DeviceType::CUDA,
        c10::DeviceType::HIP,
        c10::DeviceType::MPS,
        c10::DeviceType::MTIA,
        c10::DeviceType::XPU,
        c10::DeviceType::PrivateUse1
    );
    // Check if the device type is supported.
    if (std::find(supported_devices.begin(), supported_devices.end(), device_type) == supported_devices.end()) {
        return ErrorType::DeviceNotSupported;
    }
  switch (device_type) {
    case DeviceType::CPU: {
      // Use memory_order_relaxed here since even if two threads race,
      // they will still compute the same value for cpu_dispatch_ptr.
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      if (!fptr) {
        auto result = try_choose_cpu_impl(
          DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
          , AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
          , AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
          , VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
          , ZVECTOR
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
          , SVE512
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
          , SVE256
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
          , SVE128
#endif
        );
        if (!std::holds_alternative<ErrorType>(result)) {
          cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
        }
      return result;
      }
      return DispatchResult(fptr);
    }

    case DeviceType::CUDA:
      return cuda_dispatch_ptr != nullptr ? DispatchResult(cuda_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::HIP:
      return hip_dispatch_ptr != nullptr ? DispatchResult(hip_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_MPS)
    case DeviceType::MPS:
      return mps_dispatch_ptr != nullptr ? DispatchResult(mps_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif
    case DeviceType::MTIA:
      return mtia_dispatch_ptr != nullptr ? DispatchResult(mtia_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_XPU)
    case DeviceType::XPU:
      return xpu_dispatch_ptr != nullptr ? DispatchResult(xpu_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif

    case DeviceType::PrivateUse1:
      return privateuse1_dispatch_ptr != nullptr ? DispatchResult(privateuse1_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    default:
      TORCH_INTERNAL_ASSERT(false, "An unexpected device type was provided ", device_type);
      return ErrorType::DeviceNotSupported;
  }
}

void* DispatchStubImpl::get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
  , void *SVE512
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  , void *SVE256
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
  , void *SVE128
#endif
) {

  auto result = try_get_call_ptr(
      device_type,
      DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      ,
      VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      ,
      ZVECTOR
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
      ,
      SVE512
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
      ,
      SVE256
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
      ,
      SVE128
#endif
  );
  if (std::holds_alternative<ErrorType>(result)) {
    auto error = std::get<ErrorType>(result);
    switch (error) {
      case ErrorType::MissingDeviceKernel:
        TORCH_INTERNAL_ASSERT(
            false, "DispatchStub: missing kernel for ", device_type);
        return nullptr;
      case ErrorType::DeviceNotSupported:
        TORCH_CHECK(false, "DispatchStub: unsupported device type", device_type);
    }
  }

  void* fptr = std::get<void*>(result);
  return fptr;
}

DispatchResult DispatchStubImpl::try_choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
    , void *SVE512
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
    , void *SVE256
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
    , void *SVE128
#endif
  ){

  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
    } else {
      return DispatchResult(AVX512);
    }
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    return VSX != nullptr ? DispatchResult(VSX) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    return ZVECTOR != nullptr ? DispatchResult(ZVECTOR) : ErrorType::MissingDeviceKernel;
  }
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE512)) {
    if (C10_UNLIKELY(!SVE512)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    } else {
      return DispatchResult(SVE512);
    }
  }
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE256)) {
    if (C10_UNLIKELY(!SVE256)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    } else {
      return DispatchResult(SVE256);
    }
  }
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE128)) {
    if (C10_UNLIKELY(!SVE128)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
    } else {
      return DispatchResult(SVE128);
    }
  }
#endif
  return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
}

void* DispatchStubImpl::choose_cpu_impl(
  void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
  , void *SVE512
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  , void *SVE256
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
  , void *SVE128
#endif
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // Quantization kernels have also been disabled on Windows
    // for AVX512 because some of their tests are flaky on Windows.
    // Ideally, we should have AVX512 kernels for all kernels.
    if (C10_UNLIKELY(!AVX512)) {
      // dispatch to AVX2, since the AVX512 kernel is missing
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AVX512;
    }
  }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
    return AVX2;
  }
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    TORCH_INTERNAL_ASSERT(VSX, "DispatchStub: missing VSX kernel");
    return VSX;
  }
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    TORCH_INTERNAL_ASSERT(ZVECTOR, "DispatchStub: missing ZVECTOR kernel");
    return ZVECTOR;
  }
#endif
#ifdef HAVE_SVE512_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE512)) {
    if (C10_UNLIKELY(!SVE512)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
      return DEFAULT;
    } else {
      return SVE512;
    }
  }
#endif
#ifdef HAVE_SVE256_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE256)) {
    if (C10_UNLIKELY(!SVE256)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
      return DEFAULT;
    } else {
      return SVE256;
    }
  }
#endif
#ifdef HAVE_SVE128_CPU_DEFINITION
  if (capability >= static_cast<int>(CPUCapability::SVE128)) {
    if (C10_UNLIKELY(!SVE128)) {
      // dispatch to DEFAULT, since the SVE kernel is missing
      TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
      return DEFAULT;
    } else {
      return SVE128;
    }
  }
#endif
  TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
  return DEFAULT;
}

}  // namespace at::native
