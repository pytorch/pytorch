#include <ATen/cpu/FlushDenormal.h>
#include <ATen/cpu/vec/intrinsics.h>
#if !defined(__s390x__)
#include <cpuinfo.h>
#endif

namespace at { namespace cpu {

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
static constexpr unsigned int DENORMALS_ZERO = 0x0040;
static constexpr unsigned int FLUSH_ZERO = 0x8000;

bool set_flush_denormal(bool on) {
  // Compile if we have SSE support (GCC), x86-64 (MSVC), or x86 with SSE (MSVC)
  // Denormals-Are-Zero is supported by most SSE2 processors, with the exception
  // of some early Pentium 4 processors. We guard it with a runtime check.
  // Flush-To-Zero (FTZ) only requires SSE.
  if (cpuinfo_has_x86_daz()) {
    unsigned int csr = _mm_getcsr();
    csr &= ~DENORMALS_ZERO;
    csr &= ~FLUSH_ZERO;
    if (on) {
      csr |= DENORMALS_ZERO;
      csr |= FLUSH_ZERO;
    }
    _mm_setcsr(csr);
    return true;
  }
  return false;
}
#else
bool set_flush_denormal(bool on) {
  return false;
}
#endif

}}  // namespace at::cpu
