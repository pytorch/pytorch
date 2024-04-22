#include <ATen/cpu/Utils.h>
#if !defined(__s390x__ ) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

namespace at::cpu {

bool is_cpu_support_vnni() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
#else
  return false;
#endif
}

} // namespace at::cpu
