#include <ATen/cpu/Utils.h>
#include <cpuinfo.h>

namespace at {
namespace cpu {

bool is_cpu_support_vnni() {
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
}

} // namespace cpu
} // namespace at
