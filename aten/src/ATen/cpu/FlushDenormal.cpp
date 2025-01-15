#include <ATen/cpu/FlushDenormal.h>
#include <ATen/cpu/vec/intrinsics.h>
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

namespace at::cpu {

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
#elif defined(__ARM_FP) && (__ARM_FP > 0)
// Imported from TensorFlow, tensorflow/third_party/xla/third_party/tsl/tsl/platform/denormal.cc
// Copyright 2015 The TensorFlow Authors. All Rights Reserved.

// Flush-to-zero bit on the ARM floating-point control register.
#define ARM_FPCR_FZ   (1 << 24)

static inline void ArmSetFloatingPointControlRegister(uint32_t fpcr) {
#if defined(__aarch64__)
  __asm__ __volatile__("msr fpcr, %[fpcr]"
                       :
                       : [fpcr] "r"(static_cast<uint64_t>(fpcr)));
#else
  __asm__ __volatile__("vmsr fpscr, %[fpcr]" : : [fpcr] "r"(fpcr));
#endif
}

static inline uint32_t ArmGetFloatingPointControlRegister() {
  uint32_t fpcr;
#if defined(__aarch64__)
  uint64_t fpcr64;
  __asm__ __volatile__("mrs %[fpcr], fpcr" : [fpcr] "=r"(fpcr64));
  fpcr = static_cast<uint32_t>(fpcr64);
#else
  __asm__ __volatile__("vmrs %[fpcr], fpscr" : [fpcr] "=r"(fpcr));
#endif
  return fpcr;
}

bool set_flush_denormal(bool on) {
    uint32_t fpcr = ArmGetFloatingPointControlRegister();
    if (on) {
      fpcr |= ARM_FPCR_FZ;
    } else {
      fpcr &= ~ ARM_FPCR_FZ;
    }
    ArmSetFloatingPointControlRegister(fpcr);
    return true;
}
#else
bool set_flush_denormal(bool on) {
  return false;
}
#endif

}  // namespace at::cpu
