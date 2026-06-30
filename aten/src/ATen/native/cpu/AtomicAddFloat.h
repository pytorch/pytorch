#ifndef ATOMIC_ADD_FLOAT
#define ATOMIC_ADD_FLOAT

#include <atomic>

static inline void cpu_atomic_add_float(float* dst, float fvalue)
{
#ifdef __APPLE__
  __atomic_fetch_add(dst, fvalue, __ATOMIC_SEQ_CST);
#else
  std::atomic_ref<float>(*dst).fetch_add(fvalue);
#endif
}

#endif // ATOMIC_ADD_FLOAT
