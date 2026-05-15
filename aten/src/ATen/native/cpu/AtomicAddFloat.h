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
#if defined(__cpp_lib_atomic_ref) && __cpp_lib_atomic_ref >= 201806L
  std::atomic_ref<float> atomic_dst(*dst);
#else
  auto& atomic_dst = *reinterpret_cast<std::atomic<float>*>(dst);
#endif
  float old_value = atomic_dst.load();
  float new_value = old_value + fvalue;
  while (!atomic_dst.compare_exchange_weak(old_value, new_value)) {
#ifdef __aarch64__
    __asm__ __volatile__("yield;" : : : "memory");
#else
    _mm_pause();
#endif
    new_value = old_value + fvalue;
  }
}

#endif
