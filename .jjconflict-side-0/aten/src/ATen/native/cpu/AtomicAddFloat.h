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
#if defined(__aarch64__)
    __asm__ __volatile__("yield;" : : : "memory");
#elif defined(__riscv)
    // Zihintpause `pause` spin-loop hint, emitted as its raw encoding so it
    // assembles regardless of -march (it is a no-op HINT on cores lacking the
    // extension).
    __asm__ __volatile__(".insn i 0x0F, 0, x0, x0, 0x010" : : : "memory");
#else
    _mm_pause();
#endif
    new_value = old_value + fvalue;
  }
}

#endif
