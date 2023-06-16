#include <ATen/native/CPUUtils.h>
#include <cpuinfo.h>

namespace at {
namespace native {

bool is_cpu_support_vnni() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
}

} // namespace native
} // namespace at
