#ifndef ATOMIC_ADD_FLOAT
#define ATOMIC_ADD_FLOAT

#if (defined(__x86_64__) || defined(__i386__) || defined(__aarch64__))
#include <ATen/native/cpu/Intrinsics.h>
#else
#define _mm_pause()
#endif

#include <atomic>

static inline void cpu_atomic_add_float(float* dst, float fvalue)
{
  typedef union {
    unsigned intV;
    float floatV;
  } uf32_t;

  uf32_t new_value, old_value;
  std::atomic<unsigned>* dst_intV = (std::atomic<unsigned>*)(dst);

  old_value.floatV = *dst;
  new_value.floatV = old_value.floatV + fvalue;

  unsigned* old_intV = (unsigned*)(&old_value.intV);
  while (!std::atomic_compare_exchange_strong(dst_intV, old_intV, new_value.intV)) {
#ifdef __aarch64__
    __asm__ __volatile__("yield;" : : : "memory");
#else
    _mm_pause();
#endif
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  }
}

#endif
