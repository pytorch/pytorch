#include <ATen/cpu/FlushDenormal.h>

#include <ATen/cpu/vec256/intrinsics.h>
#include <cpuinfo.h>

namespace at { namespace cpu {

static constexpr unsigned int DENORMALS_ZERO = 0x0040;
static constexpr unsigned int FLUSH_ZERO = 0x8000;

bool set_flush_denormal(bool on) {
  // Compile if we have SSE support (GCC), x86-64 (MSVC), or x86 with SSE (MSVC)
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
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
#endif
  return false;
}

}}  // namespace at::cpu
