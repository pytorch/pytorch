#include <ATen/cpu/Utils.h>
#if !defined(__s390x__)
#include <cpuinfo.h>
#endif

namespace at {
namespace cpu {

bool is_cpu_support_vnni() {
#if !defined(__s390x__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
#else
  return false;
#endif
}

} // namespace cpu
} // namespace at
