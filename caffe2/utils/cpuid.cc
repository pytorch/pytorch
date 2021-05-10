#include "caffe2/utils/cpuid.h"

namespace caffe2 {

const CpuId& GetCpuId() {
  static CpuId cpuid_singleton;
  return cpuid_singleton;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API uint32_t CpuId::f1c_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API uint32_t CpuId::f1d_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API uint32_t CpuId::f7b_ = 0;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TORCH_API uint32_t CpuId::f7c_ = 0;

CpuId::CpuId() {
#ifdef _MSC_VER
  int reg[4];
  __cpuid(static_cast<int*>(reg), 0);
  const int n = reg[0];
  if (n >= 1) {
    __cpuid(static_cast<int*>(reg), 1);
    f1c_ = uint32_t(reg[2]);
    f1d_ = uint32_t(reg[3]);
  }
  if (n >= 7) {
    __cpuidex(static_cast<int*>(reg), 7, 0);
    f7b_ = uint32_t(reg[1]);
    f7c_ = uint32_t(reg[2]);
  }
#elif defined(__i386__) && defined(__PIC__) && !defined(__clang__) && \
    defined(__GNUC__)
  // The following block like the normal cpuid branch below, but gcc
  // reserves ebx for use of its pic register so we must specially
  // handle the save and restore to avoid clobbering the register
  uint32_t n;
  __asm__(
      "pushl %%ebx\n\t"
      "cpuid\n\t"
      "popl %%ebx\n\t"
      : "=a"(n)
      : "a"(0)
      : "ecx", "edx");
  if (n >= 1) {
    uint32_t f1a;
    __asm__(
        "pushl %%ebx\n\t"
        "cpuid\n\t"
        "popl %%ebx\n\t"
        : "=a"(f1a), "=c"(f1c_), "=d"(f1d_)
        : "a"(1)
        :);
  }
  if (n >= 7) {
    __asm__(
        "pushl %%ebx\n\t"
        "cpuid\n\t"
        "movl %%ebx, %%eax\n\r"
        "popl %%ebx"
        : "=a"(f7b_), "=c"(f7c_)
        : "a"(7), "c"(0)
        : "edx");
  }
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t n;
  __asm__("cpuid" : "=a"(n) : "a"(0) : "ebx", "ecx", "edx");
  if (n >= 1) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint32_t f1a;
    __asm__("cpuid" : "=a"(f1a), "=c"(f1c_), "=d"(f1d_) : "a"(1) : "ebx");
  }
  if (n >= 7) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint32_t f7a;
    __asm__("cpuid"
            : "=a"(f7a), "=b"(f7b_), "=c"(f7c_)
            : "a"(7), "c"(0)
            : "edx");
  }
#endif
}

} // namespace caffe2
