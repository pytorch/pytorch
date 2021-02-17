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

}}  // namespace at::native
