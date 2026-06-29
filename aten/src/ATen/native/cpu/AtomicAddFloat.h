#ifndef ATOMIC_ADD_FLOAT
#define ATOMIC_ADD_FLOAT

#include <atomic>

static inline void cpu_atomic_add_float(float* dst, float fvalue)
{
  std::atomic_ref<float>(*dst).fetch_add(fvalue);
}

#endif // ATOMIC_ADD_FLOAT
