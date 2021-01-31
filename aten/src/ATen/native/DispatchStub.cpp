#include <ATen/native/DispatchStub.h>

#include <c10/util/Exception.h>

#include <cpuinfo.h>
#include <cstdlib>
#include <cstring>

namespace at { namespace native {

static CPUCapability compute_cpu_capability() {
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
#ifdef HAVE_VSX_CPU_DEFINITION
    if (strcmp(envar, "vsx") == 0) {
      return CPUCapability::VSX;
    }
#else
    if (strcmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }
    if (strcmp(envar, "avx") == 0) {
      return CPUCapability::AVX;
    }
#endif
    if (strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
  }

#if !defined(__powerpc__) && !defined(__s390x__)
  if (cpuinfo_initialize()) {
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX2;
    }
    if (cpuinfo_has_x86_avx()) {
      return CPUCapability::AVX;
    }
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

DispatchStub::FnPtr DispatchStub::get_call_ptr(DeviceType device_type) {
  FnPtr call_ptr = nullptr;
  if (device_type == DeviceType::CPU) {
    // Use memory_order_relaxed here since even if two threads race,
    // they will still compute the same value for cpu_dispatch_ptr.
    auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
    if (!fptr) {
      fptr = choose_cpu_impl();
      cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
    }
    call_ptr = fptr;
  } else if (device_type == DeviceType::CUDA) {
    AT_ASSERTM(cuda_dispatch_ptr, "DispatchStub: missing CUDA kernel");
    call_ptr = cuda_dispatch_ptr;
  } else if (device_type == DeviceType::HIP) {
    AT_ASSERTM(hip_dispatch_ptr, "DispatchStub: missing HIP kernel");
    call_ptr = hip_dispatch_ptr;
  }
  if (call_ptr == nullptr) {
    AT_ERROR("DispatchStub: unsupported device type", device_type);
  }
  return call_ptr;
}

}}  // namespace at::native
