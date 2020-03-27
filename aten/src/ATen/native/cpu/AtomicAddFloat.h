#ifndef ATOMIC_ADD_FLOAT
#define ATOMIC_ADD_FLOAT

#include <immintrin.h>
#include <atomic>

template <typename T, typename U>
static inline void cpu_atomic_add(T* dst, T fvalue)
{
  typedef union {
    U uintV;
    T floatV;
  } uf_t;

  uf_t new_value, old_value;
  std::atomic<U>* dst_uintV = (std::atomic<U>*)(dst);

  old_value.floatV = *dst;
  new_value.floatV = old_value.floatV + fvalue;

  U* old_uintV = (U*)(&old_value.uintV);
  while (!std::atomic_compare_exchange_strong(dst_uintV, old_uintV, new_value.uintV)) {
    _mm_pause();
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  }
}

#endif
