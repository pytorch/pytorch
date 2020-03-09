#pragma once

#include <stdint.h>

#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

struct fpu_state {
#if defined(__SSE__) || defined(__x86_64__)
  uint32_t mxcsr;
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  uint32_t fpscr;
#elif defined(__aarch64__)
  uint64_t fpcr;
#else
  char unused;
#endif
};

static inline struct fpu_state get_fpu_state() {
  struct fpu_state state = { 0 };
#if defined(__SSE__) || defined(__x86_64__)
  state.mxcsr = (uint32_t) _mm_getcsr();
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  __asm__ __volatile__("VMRS %[fpscr], fpscr" : [fpscr] "=r" (state.fpscr));
#elif defined(__aarch64__)
  __asm__ __volatile__("MRS %[fpcr], fpcr" : [fpcr] "=r" (state.fpcr));
#endif
  return state;
}

static inline void set_fpu_state(const struct fpu_state state) {
#if defined(__SSE__) || defined(__x86_64__)
  _mm_setcsr((unsigned int) state.mxcsr);
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  __asm__ __volatile__("VMSR fpscr, %[fpscr]" : : [fpscr] "r" (state.fpscr));
#elif defined(__aarch64__)
  __asm__ __volatile__("MSR fpcr, %[fpcr]" : : [fpcr] "r" (state.fpcr));
#endif
}

static inline void disable_fpu_denormals() {
#if defined(__SSE__) || defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | 0x8040);
#elif defined(__arm__) && defined(__ARM_FP) && (__ARM_FP != 0)
  uint32_t fpscr;
  __asm__ __volatile__(
      "VMRS %[fpscr], fpscr\n"
      "ORR %[fpscr], #0x1000000\n"
      "VMSR fpscr, %[fpscr]\n"
    : [fpscr] "=r" (fpscr));
#elif defined(__aarch64__)
  uint64_t fpcr;
  __asm__ __volatile__(
      "MRS %[fpcr], fpcr\n"
      "ORR %w[fpcr], %w[fpcr], 0x1000000\n"
      "ORR %w[fpcr], %w[fpcr], 0x80000\n"
      "MSR fpcr, %[fpcr]\n"
    : [fpcr] "=r" (fpcr));
#endif
}
